# api/routes/hackrx.py

import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import time
import logging
from urllib.parse import urlparse
import httpx

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
from api.core.document_processor import stream_document, process_document_stream, optimized_semantic_chunk_text, build_enhanced_retrieval_systems
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_orchestrator, prepare_query_strategies_for_all_questions, answer_image_query
from api.core.embedding_manager import OptimizedEmbeddingManager
import api.core.agent_logic as agent_logic

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

# --- Custom exception for clearer error handling ---
class UnsupportedFileType(Exception):
    pass

def validate_file_type(url: str) -> None:
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower() or ".txt"
    supported_formats = ['.pdf', '.docx', '.xlsx', '.txt', '.md', '.csv', '.pptx', '.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif', '.webp']
    if file_ext not in supported_formats:
        raise UnsupportedFileType(f"Unsupported file type: '{file_ext}'. Supported formats are: {', '.join(supported_formats)}")

async def process_document_and_build_kb(document_url: str, manager: OptimizedEmbeddingManager, enable_enhanced_retrieval: bool) -> Tuple[RequestKnowledgeBase, bool]:
    """
    SIMPLIFIED: This function no longer builds the multi-vector system.
    'enhanced_available' now only refers to the query expander.
    """
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
    enhanced_available = False
    
    if chunks:
        await knowledge_base.build(chunks, precomputed_embeddings)
        if enable_enhanced_retrieval:
            # Only build the query expander
            query_exp = await build_enhanced_retrieval_systems(chunks)
            agent_logic.query_expander = query_exp
            enhanced_available = query_exp is not None
            if enhanced_available:
                print("✨ Query expansion system successfully initialized")
    else:
        print("PIPE-DOC: ⚠️ No chunks were generated from the document.")
    
    t5 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ KB indexing (+ query expander) took: {t5 - t4:.2f}s")
    print(f"PIPE-DOC: ✅ Full document pipeline complete in {time.perf_counter() - pipeline_start_time:.2f}s.")
    return knowledge_base, enhanced_available

@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    start_time = time.perf_counter()
    try:
        validate_file_type(request.documents)
        print(f"✅ File type validation passed for: {request.documents}")
        
        if is_image_url(request.documents):
            print("🖼️ Image URL detected. Using direct vision query.")
            async with httpx.AsyncClient() as client:
                response = await client.get(request.documents, follow_redirects=True, timeout=60.0)
                response.raise_for_status()
                image_bytes = response.content
            tasks = [answer_image_query(image_bytes, q) for q in request.questions]
            final_answers = await asyncio.gather(*tasks)
        else:
            print("📄 Document URL detected. Starting RAG pipeline.")
            manager = ml_models.get("embedding_manager")
            if not manager:
                raise HTTPException(status_code=503, detail="Embedding manager not ready.")
            
            doc_pipeline_task = process_document_and_build_kb(request.documents, manager, enable_enhanced_retrieval=True)
            query_strategy_task = prepare_query_strategies_for_all_questions(request.questions)
            (knowledge_base, enhanced_available), query_strategy_data_list = await asyncio.gather(doc_pipeline_task, query_strategy_task)

            if not knowledge_base.chunks:
                return RunResponse(answers=["Document appears to be empty or unparsable."] * len(request.questions))

            tasks = [
                answer_question_orchestrator(knowledge_base, data, use_high_k=len(request.questions) <= 15, use_enhanced_retrieval=enhanced_available)
                for data in query_strategy_data_list
            ]
            results_with_context = await asyncio.gather(*tasks)
            final_answers = [ans for ans, ctx in results_with_context]

        for question, answer in zip(request.questions, final_answers):
            qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
        
        print(f"🏁 Total request processing time: {time.perf_counter() - start_time:.2f} seconds")
        return RunResponse(answers=final_answers)

    except UnsupportedFileType as e:
        error_message = f"Filetype not supported. {e}"
        print(f"🚫 {error_message}")
        return RunResponse(answers=[error_message for _ in request.questions])
    except Exception as e:
        logging.error(f"An internal error occurred during /run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")