import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import time
import logging
from urllib.parse import urlparse
import httpx
from langdetect import detect

# --- Logger Configuration ---
qa_logger = logging.getLogger("qa_logger")
if not qa_logger.handlers:
    qa_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("qa_log.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - Q: %(message)s")
    file_handler.setFormatter(formatter)
    qa_logger.addHandler(file_handler)
    qa_logger.propagate = False

# --- Core Logic Imports ---
from api.state import ml_models
from api.core.document_processor import (
    stream_document,
    process_document_stream,
    optimized_semantic_chunk_text,
)
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import (
    answer_raw_text_query,
    prepare_query_strategies_for_all_questions,
    answer_image_query,
    answer_questions_batch_orchestrator,
    synthesize_direct_answer,
)
from api.core.embedding_manager import OptimizedEmbeddingManager

# --- New Agno Agent Import ---
from api.core.agno_agent import (
    process_with_agno_agent_simple,
    should_use_direct_processing,
)

# --- API Router and Pydantic Models ---
hackrx_router = APIRouter(prefix="/hackrx")


class RunRequest(BaseModel):
    documents: str = Field(
        ..., description="URL of the PDF, DOCX, image, or text document to process."
    )
    questions: List[str] = Field(
        ..., description="A list of questions to answer based on the document."
    )


class RunResponse(BaseModel):
    answers: List[str]


# --- Authentication ---
auth_scheme = HTTPBearer()
EXPECTED_TOKEN = "7bf4409966a1479a8578f3258eba4e215cef0f7ccd694a2440149c1eeb4874ef"


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if (
        not credentials
        or credentials.scheme != "Bearer"
        or credentials.credentials != EXPECTED_TOKEN
    ):
        raise HTTPException(
            status_code=401, detail="Invalid or missing authentication token"
        )


def is_image_url(url: str) -> bool:
    image_formats = [".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".gif", ".webp"]
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower()
    return file_ext in image_formats


class UnsupportedFileType(Exception):
    pass


def validate_file_type(url: str) -> None:
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower() or ".txt"
    supported_formats = [
        ".pdf",
        ".docx",
        ".xlsx",
        ".txt",
        ".md",
        ".csv",
        ".pptx",
        ".png",
        ".jpeg",
        ".jpg",
        ".bmp",
        ".tiff",
        ".gif",
        ".webp",
    ]
    if file_ext not in supported_formats:
        raise UnsupportedFileType(
            f"The file format '{file_ext}' is not supported. "
            f"Please upload a valid document or image file (e.g., {', '.join(supported_formats)})."
        )


# --- Language Utility ---
def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return True


def is_raw_text_url(url: str) -> bool:
    """Detect URLs that are not known file formats and should be treated as raw text/HTML."""
    known_exts = [
        ".pdf",
        ".docx",
        ".xlsx",
        ".txt",
        ".md",
        ".csv",
        ".pptx",
        ".png",
        ".jpeg",
        ".jpg",
        ".bmp",
        ".tiff",
        ".gif",
        ".webp",
    ]
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower()
    return file_ext not in known_exts


# --- Document KB builder ---
async def process_document_and_build_kb(
    document_url: str, manager: OptimizedEmbeddingManager
) -> RequestKnowledgeBase:
    pipeline_start_time = time.perf_counter()
    print(f"PIPE-DOC: Starting document pipeline for: {document_url}")

    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    pages_data = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Download & text extraction took: {t1 - t0:.2f}s")

    t2 = time.perf_counter()
    chunks, precomputed_embeddings = await optimized_semantic_chunk_text(
        pages_data, manager
    )
    t3 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Semantic chunking (incl. embedding) took: {t3 - t2:.2f}s")

    t4 = time.perf_counter()
    knowledge_base = RequestKnowledgeBase(manager)

    if chunks:
        await knowledge_base.build(chunks, precomputed_embeddings)
    else:
        print("PIPE-DOC: ⚠️ No chunks were generated from the document.")

    t5 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ KB indexing took: {t5 - t4:.2f}s")
    print(
        f"PIPE-DOC: ✅ Full document pipeline complete in {time.perf_counter() - pipeline_start_time:.2f}s."
    )
    return knowledge_base


# --- /run Endpoint with restored RAG logic ---
@hackrx_router.post(
    "/run", response_model=RunResponse, dependencies=[Depends(verify_token)]
)
async def run_submission(request: RunRequest = Body(...)):
    try:
        start = time.perf_counter()
        validate_file_type(request.documents)

        if is_raw_text_url(request.documents):
            print("📝 Raw text/HTML detected: fetching and sending directly to LLM.")
            async with httpx.AsyncClient() as client:
                resp = await client.get(request.documents, timeout=60.0)
                resp.raise_for_status()
                raw_text = resp.text.strip()

            answers = await asyncio.gather(
                *[answer_raw_text_query(raw_text, q) for q in request.questions]
            )
            print(f"⏱️ Total time (raw text): {time.perf_counter() - start:.2f}s")
            for question, answer in zip(request.questions, answers):
                try:
                    qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
                except Exception as e:
                    print(f"⚠️ Error logging QA pair: {e}")
            return RunResponse(answers=answers)

        if is_image_url(request.documents):
            print("🖼️ Image URL detected. Using direct vision query.")
            # No need to download the image, pass URL directly
            answers = await asyncio.gather(
                *[answer_image_query(request.documents, q) for q in request.questions]
            )
            print(f"⏱️ Total time (image): {time.perf_counter() - start:.2f}s")
            for question, answer in zip(request.questions, answers):
                try:
                    qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
                except Exception as e:
                    print(f"⚠️ Error logging QA pair: {e}")
            return RunResponse(answers=answers)

        # Document processing starts here
        doc_iter = stream_document(request.documents)
        pages_data = await process_document_stream(request.documents, doc_iter)
        full_text = "\n\n".join(p for p, _ in pages_data).strip()

        doc_sample = full_text[:1000]
        doc_en = is_english(doc_sample)
        qs_en = all(is_english(q) for q in request.questions)
        use_high_k = len(request.questions) <= 18
        use_direct_translate = should_use_direct_processing(full_text, token_limit=2000)

        if not doc_en or not qs_en:
            print(
                "🌐 Non-English detected: running synthesis on-device for all questions in parallel."
            )
            synthesis_tasks = [
                synthesize_direct_answer(q, full_text, True) for q in request.questions
            ]
            answers = await asyncio.gather(*synthesis_tasks)
            print(f"⏱️ Total time (non-English): {time.perf_counter() - start:.2f}s")
            for question, answer in zip(request.questions, answers):
                try:
                    qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
                except Exception as e:
                    print(f"⚠️ Error logging QA pair: {e}")
            return RunResponse(answers=answers)

        file_exts = [".pdf", ".docx", ".pptx", ".txt"]
        is_supported_doc = any(
            ext in str(request.documents).lower() for ext in file_exts
        )
        use_direct_agno = use_direct_translate and is_supported_doc

        if use_direct_agno:
            print(
                f"📄 Document under 2000 tokens - using Agno agent for direct processing"
            )
            try:
                answers = await process_with_agno_agent_simple(
                    request.documents, request.questions, full_text
                )
                print(f"⏱️ Total time (Agno direct): {time.perf_counter() - start:.2f}s")
                for question, answer in zip(request.questions, answers):
                    try:
                        qa_logger.info(
                            f"{question} | A: {answer.replace(chr(10), ' ')}"
                        )
                    except Exception as e:
                        print(f"⚠️ Error logging QA pair: {e}")
                return RunResponse(answers=answers)
            except Exception as e:
                print(
                    f"❌ Agno processing failed: {e}, falling back to full RAG pipeline"
                )

        # --- MERGED LOGIC: Full RAG pipeline with Query Decomposition and Batching ---
        print(
            f"📚 Document over 2000 tokens - using full RAG pipeline with query decomposition"
        )
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(503, "Embedding manager not ready.")

        # Concurrently build the knowledge base AND prepare the query strategies
        doc_pipeline_task = process_document_and_build_kb(request.documents, manager)
        query_strategy_task = prepare_query_strategies_for_all_questions(
            request.questions
        )
        knowledge_base, query_strategy_data_list = await asyncio.gather(
            doc_pipeline_task, query_strategy_task
        )

        if not knowledge_base.chunks:
            return RunResponse(
                answers=["Document appears to be empty or unparsable."]
                * len(request.questions)
            )

        # Use the batched orchestrator which handles decomposition, RRF, and batched reranking
        results_with_context = await answer_questions_batch_orchestrator(
            knowledge_base, query_strategy_data_list, use_high_k=use_high_k
        )
        answers = [ans for ans, _ in results_with_context]

        print(f"⏱️ Total time (RAG): {time.perf_counter() - start:.2f}s")
        for question, answer in zip(request.questions, answers):
            qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
        return RunResponse(answers=answers)

    except UnsupportedFileType as e:
        error_message = f"Filetype not supported. {e}"
        print(f"🚫 {error_message}")
        return JSONResponse(content={"error": error_message}, status_code=200)

    except Exception as e:
        logging.error(f"An internal error occurred during /run: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )
