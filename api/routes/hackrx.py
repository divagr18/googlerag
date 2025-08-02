# api/routes/hackrx.py
import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import time
import logging

# --- Logger Configuration (Correct and Unchanged) ---
qa_logger = logging.getLogger('qa_logger')
qa_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('qa_log.log', mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
if not qa_logger.handlers:
    qa_logger.addHandler(file_handler)

# --- Imports (Unchanged) ---
from api.state import ml_models
from api.core.document_processor import stream_document, process_document_stream, optimized_semantic_chunk_text
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_with_agent, prepare_enhanced_queries_for_all_questions
from api.core.embedding_manager import OptimizedEmbeddingManager

# --- Router and Models (Unchanged) ---
hackrx_router = APIRouter(prefix="/hackrx")

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

auth_scheme = HTTPBearer()
EXPECTED_TOKEN = "7bf4409966a1479a8578f3258eba4e215cef0f7ccd694a2440149c1eeb4874ef"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

async def process_document_and_build_kb(
    document_url: str, 
    manager: OptimizedEmbeddingManager
) -> RequestKnowledgeBase:
    """
    A single awaitable task that encapsulates the entire document processing pipeline,
    using streaming to overlap download and processing.
    """
    pipeline_start_time = time.perf_counter()
    print(f"PIPE-DOC: Starting full document pipeline for: {document_url}")
    
    # Step 1: Stream the document and process it in parallel
    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    document_text = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Streamed download & text extraction took: {t1 - t0:.2f}s")
    
    # Step 2: Chunk and get pre-computed embeddings
    t2 = time.perf_counter()
    chunks, precomputed_embeddings = await optimized_semantic_chunk_text(document_text, manager)
    t3 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Semantic chunking (incl. embedding) took: {t3 - t2:.2f}s")
    
    # Step 3: Build Knowledge Base
    t4 = time.perf_counter()
    knowledge_base = RequestKnowledgeBase(manager)
    if chunks:
        await knowledge_base.build(chunks, precomputed_embeddings)
    t5 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ KB indexing (BM25 + FAISS) took: {t5 - t4:.2f}s")
    
    pipeline_end_time = time.perf_counter()
    print(f"PIPE-DOC: ✅ Full document pipeline complete in {pipeline_end_time - pipeline_start_time:.2f}s.")
    return knowledge_base

@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    """
    Processes a document and answers questions using a streamlined, optimized pipeline with parallel processing.
    """
    start_time = time.perf_counter()

    manager = ml_models.get("embedding_manager")
    if not manager:
        raise HTTPException(status_code=503, detail="Embedding manager is not ready.")

    try:
        # --- Phase 1: MAXIMALLY PARALLEL PREPARATION ---
        t0 = time.perf_counter()
        print("🚀 Starting maximally parallel preparation: Document Pipeline vs. Query Enhancement")

        # Create the two independent, top-level tasks
        doc_pipeline_task = process_document_and_build_kb(request.documents, manager)
        query_enhancement_task = prepare_enhanced_queries_for_all_questions(request.questions)
        
        # Run them concurrently and wait for both to finish
        knowledge_base, enhanced_query_data = await asyncio.gather(
            doc_pipeline_task, 
            query_enhancement_task
        )
        
        t1 = time.perf_counter()
        print(f"✅ Parallel preparation completed in {t1 - t0:.2f} seconds.")

        # --- Safety check for empty document ---
        if not knowledge_base.chunks:
            print("⚠️ Document processing resulted in zero chunks. Cannot answer questions.")
            answers = ["I could not find relevant information in the document because it appears to be empty or unparsable."] * len(request.questions)
            return RunResponse(answers=answers)

        # --- Phase 2: Optimized Agent Execution (Unchanged) ---
        t2 = time.perf_counter()
        
        tasks = [
            answer_question_with_agent(question, knowledge_base, precomputed_data)
            for i, question in enumerate(request.questions)
            for precomputed_data in [enhanced_query_data[i]]
        ]
        
        print(f"🎯 Spawning {len(request.questions)} optimized agents...")
        answers = await asyncio.gather(*tasks)
        print("✅ All agent tasks completed.")
        t3 = time.perf_counter()
        print(f"⚡️ Agent Execution took: {t3 - t2:.4f} seconds.")

        # --- Phase 3: Logging and Response (Unchanged) ---
        print("Logging Q&A pairs to qa_log.log...")
        for question, answer in zip(request.questions, answers):
            cleaned_answer = answer.replace('\n', ' ')
            qa_logger.info(f"Q: {question} | A: {cleaned_answer}")
        
        end = time.perf_counter()
        print(f"🏁 Total processing time: {end - start_time:.2f} seconds")
        print(f"🏆 Performance boost achieved through maximum parallelization!")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        qa_logger.error(f"An internal error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")