# api/routes/hackrx.py

import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import time
import logging
from urllib.parse import urlparse
# --- NEW: Import httpx to download the image directly ---
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
from api.core.document_processor import stream_document, process_document_stream, optimized_semantic_chunk_text
from api.core.vector_store import RequestKnowledgeBase
# --- NEW: Import the new image query function ---
from api.core.agent_logic import answer_question_orchestrator, prepare_query_strategies_for_all_questions, answer_image_query
from api.core.embedding_manager import OptimizedEmbeddingManager
from agno.models.google import Gemini

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
    return credentials

# --- NEW: Helper function to check if the URL points to an image ---
def is_image_url(url: str) -> bool:
    """Checks the file extension of a URL to see if it's a common image format."""
    image_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif', '.webp']
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower()
    return file_ext in image_formats

# --- Helper Function for Document Processing Pipeline ---
async def process_document_and_build_kb(
    document_url: str, 
    manager: OptimizedEmbeddingManager
) -> RequestKnowledgeBase:
    # This function remains unchanged
    pipeline_start_time = time.perf_counter()
    print(f"PIPE-DOC: Starting full document pipeline for: {document_url}")
    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    pages_data = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Streamed download & text extraction took: {t1 - t0:.2f}s")
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
    print(f"PIPE-DOC: ⏱️ KB indexing (BM25 + FAISS) took: {t5 - t4:.2f}s")
    pipeline_end_time = time.perf_counter()
    print(f"PIPE-DOC: ✅ Full document pipeline complete in {pipeline_end_time - pipeline_start_time:.2f}s.")
    return knowledge_base

# --- Main API Endpoint ---
@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    start_time = time.perf_counter()
    final_answers = []

    try:
        # --- NEW: ROUTER LOGIC ---
        # Check if the document URL is an image.
        if is_image_url(request.documents):
            print("🖼️ Image URL detected. Bypassing RAG pipeline for direct vision query.")
            
            # Download the image bytes once.
            async with httpx.AsyncClient() as client:
                response = await client.get(request.documents, follow_redirects=True, timeout=60.0)
                response.raise_for_status()
                image_bytes = response.content

            # Create a task for each question using the new image query function.
            tasks = [
                answer_image_query(image_bytes, q) for q in request.questions
            ]
            final_answers = await asyncio.gather(*tasks)

        else:
            # --- EXISTING RAG PIPELINE LOGIC ---
            print("📄 Document URL detected. Starting RAG pipeline.")
            manager = ml_models.get("embedding_manager")
            if not manager:
                raise HTTPException(status_code=503, detail="Embedding manager is not ready.")
            
            # Phase 1: Parallel Preparation
            t0 = time.perf_counter()
            doc_pipeline_task = process_document_and_build_kb(request.documents, manager)
            query_strategy_task = prepare_query_strategies_for_all_questions(request.questions)
            knowledge_base, query_strategy_data_list = await asyncio.gather(doc_pipeline_task, query_strategy_task)
            t1 = time.perf_counter()
            print(f"✅ Parallel preparation completed in {t1 - t0:.2f} seconds.")

            if not knowledge_base.chunks:
                answers = ["I could not find relevant information in the document because it appears to be empty or unparsable."] * len(request.questions)
                return RunResponse(answers=answers)

            # Phase 2: Orchestration
            num_questions = len(request.questions)
            use_high_k = num_questions <= 15
            tasks = [
                answer_question_orchestrator(knowledge_base, data, use_high_k)
                for data in query_strategy_data_list
            ]
            results_with_context = await asyncio.gather(*tasks)
            final_answers = [ans for ans, ctx in results_with_context]

        # --- Phase 3: Logging and Response (Common to both paths) ---
        print("Logging Q&A pairs to qa_log.log...")
        for question, answer in zip(request.questions, final_answers):
            cleaned_answer = answer.replace('\n', ' ').replace('\r', '')
            qa_logger.info(f"{question} | A: {cleaned_answer}")
        
        end_time = time.perf_counter()
        print(f"🏁 Total request processing time: {end_time - start_time:.2f} seconds")
        return RunResponse(answers=final_answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An internal error occurred during /run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")