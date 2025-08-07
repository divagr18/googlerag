# api/routes/hackrx.py

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
import tiktoken  # for accurate token counting

# --- Logger Configuration ---
qa_logger = logging.getLogger('qa_logger')
if not qa_logger.handlers:
    qa_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('qa_log.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - Q: %(message)s')
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
    answer_question_orchestrator,
    prepare_query_strategies_for_all_questions,
    answer_image_query,
    answer_questions_batch_orchestrator,
    synthesize_answer_from_context,
)
from api.core.embedding_manager import OptimizedEmbeddingManager

# --- API Router and Pydantic Models ---
hackrx_router = APIRouter(prefix="/hackrx")

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF, DOCX, image, or text document to process.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
auth_scheme = HTTPBearer()
EXPECTED_TOKEN = "7bf4409966a1479a8578f3258eba4e215cef0f7ccd694a2440149c1eeb4874ef"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

def is_image_url(url: str) -> bool:
    image_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif', '.webp']
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower()
    return file_ext in image_formats

class UnsupportedFileType(Exception):
    pass

def validate_file_type(url: str) -> None:
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower() or ".txt"
    supported_formats = [
        '.pdf', '.docx', '.xlsx', '.txt', '.md', '.csv',
        '.pptx', '.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif', '.webp'
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

# --- Document KB builder (unchanged) ---
async def process_document_and_build_kb(document_url: str, manager: OptimizedEmbeddingManager) -> RequestKnowledgeBase:
    pipeline_start_time = time.perf_counter()
    print(f"PIPE-DOC: Starting document pipeline for: {document_url}")
    
    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    pages_data = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Download & text extraction took: {t1 - t0:.2f}s")
    
    t2 = time.perf_counter()
    chunks, precomputed_embeddings = await optimized_semantic_chunk_text(pages_data, manager)
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
    print(f"PIPE-DOC: ✅ Full document pipeline complete in {time.perf_counter() - pipeline_start_time:.2f}s.")
    return knowledge_base

# --- /run Endpoint with token-limit & foreign-language handling ---
@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    start = time.perf_counter()

    # 1) Validate
    validate_file_type(request.documents)

    # 2) Load & extract raw pages
    doc_iter = stream_document(request.documents)
    pages_data = await process_document_stream(request.documents, doc_iter)
    full_text = "\n\n".join(p for p,_ in pages_data).strip()

    # 3) Token‐count check
    enc = tiktoken.encoding_for_model("gpt-4")
    if len(enc.encode(full_text)) > 4000:
        raise HTTPException(400, "Document exceeds 4000‐token limit.")

    # 4) Lang detect on doc + all questions
    doc_sample = full_text[:1000]
    doc_en = is_english(doc_sample)
    qs_en = all(is_english(q) for q in request.questions)
    use_high_k = len(request.questions) <= 18

    # 5) Non‐English shortcut: direct LLM calls
    if not doc_en or not qs_en:
        print("🌐 Non-English detected: running synthesis on-device for all questions in parallel.")
        # Create a synthesis task for each question
        synthesis_tasks = [
            synthesize_answer_from_context(q, full_text, False)
            for q in request.questions
        ]
        # Run them concurrently
        answers = await asyncio.gather(*synthesis_tasks)
        return RunResponse(answers=answers)

    # 6) English & image?
    if is_image_url(request.documents):
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.documents, timeout=60.0)
            resp.raise_for_status()
            img_bytes = resp.content
        answers = await asyncio.gather(*[
            answer_image_query(img_bytes, q) for q in request.questions
        ])
    else:
        # 7) Full RAG pipeline
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(503, "Embedding manager not ready.")
        
        # build KB + chunk/embeds
        kb = await process_document_and_build_kb(request.documents, manager)

        if not kb.chunks:
            return RunResponse(answers=["Document empty or unparsable."] * len(request.questions))

        # Directly use original questions (no query generation)
        results = await asyncio.gather(*[
            answer_question_orchestrator(
                kb,
                {"original_question": q, "sub_questions": [q]},
                use_high_k=use_high_k
            )
            for q in request.questions
        ])
        answers = [ans for ans, _ in results]

    # 8) Log & return
    for q,a in zip(request.questions, answers):
        qa_logger.info(f"{q} | A: {a.replace(chr(10),' ')}")
    print(f"⏱️ Total time: {time.perf_counter()-start:.2f}s")
    return RunResponse(answers=answers)