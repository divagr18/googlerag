# api/routes/hackrx.py

import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import time
import logging

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
from api.core.agent_logic import answer_question_orchestrator, prepare_query_strategies_for_all_questions
from api.core.embedding_manager import OptimizedEmbeddingManager
from agno.models.google import Gemini

# --- API Router and Pydantic Models ---
hackrx_router = APIRouter(prefix="/hackrx")

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF, DOCX, or text document to process.")
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

# --- Helper Function for Document Processing Pipeline ---
async def process_document_and_build_kb(
    document_url: str, 
    manager: OptimizedEmbeddingManager
) -> RequestKnowledgeBase:
    pipeline_start_time = time.perf_counter()
    print(f"PIPE-DOC: Starting full document pipeline for: {document_url}")
    
    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    document_text = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Streamed download & text extraction took: {t1 - t0:.2f}s")
    
    t2 = time.perf_counter()
    chunks, precomputed_embeddings = await optimized_semantic_chunk_text(document_text, manager)
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

    manager = ml_models.get("embedding_manager")
    if not manager:
        raise HTTPException(status_code=503, detail="Embedding manager is not ready.")
        
    if "llm" not in ml_models:
        ml_models["llm"] = Gemini(id="gemini-2.5-flash", temperature=0.1, api_key=os.getenv("GOOGLE_API_KEY"))

    try:
        # --- Phase 1: MAXIMALLY PARALLEL PREPARATION ---
        t0 = time.perf_counter()
        print("🚀 Starting parallel preparation: Document Pipeline vs. Query Strategy Generation")

        doc_pipeline_task = process_document_and_build_kb(request.documents, manager)
        query_strategy_task = prepare_query_strategies_for_all_questions(request.questions)
        
        knowledge_base, query_strategy_data_list = await asyncio.gather(
            doc_pipeline_task, 
            query_strategy_task
        )
        
        t1 = time.perf_counter()
        print(f"✅ Parallel preparation completed in {t1 - t0:.2f} seconds.")

        if not knowledge_base.chunks:
            print("⚠️ Document processing resulted in zero chunks. Cannot answer questions.")
            answers = ["I could not find relevant information in the document because it appears to be empty or unparsable."] * len(request.questions)
            return RunResponse(answers=answers)

        # --- Phase 2: Optimized Orchestration ---
        t2 = time.perf_counter()
        
        # FIX: Determine if dynamic k should be high or low based on question count.
        num_questions = len(request.questions)
        use_high_k = num_questions <= 15
        if use_high_k:
            print(f"✅ {num_questions} questions (<=15). Using high-K values for quality.")
        else:
            print(f"⚠️ {num_questions} questions (>15). Using low-K values for speed.")

        # FIX: Pass the use_high_k flag to each orchestrator task.
        tasks = [
            answer_question_orchestrator(
                knowledge_base, 
                strategy_data,
                use_high_k=use_high_k
            )
            for strategy_data in query_strategy_data_list
        ]
        
        print(f"🎯 Spawning {len(tasks)} retrieval/synthesis orchestrations...")
        answers = await asyncio.gather(*tasks)
        print("✅ All orchestration tasks completed.")
        t3 = time.perf_counter()
        print(f"⚡️ Orchestration & Synthesis took: {t3 - t2:.4f} seconds.")

        # --- Phase 3: Logging and Response ---
        print("Logging Q&A pairs to qa_log.log...")
        for question, answer in zip(request.questions, answers):
            cleaned_answer = answer.replace('\n', ' ').replace('\r', '')
            qa_logger.info(f"{question} | A: {cleaned_answer}")
        
        end_time = time.perf_counter()
        print(f"🏁 Total request processing time: {end_time - start_time:.2f} seconds")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"An internal error occurred during /run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")