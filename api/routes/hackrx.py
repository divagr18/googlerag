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
from api.core.document_processor import download_document, optimized_semantic_chunk_text, process_document
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_with_agent

# --- Router and Models (Unchanged) ---
hackrx_router = APIRouter(prefix="/hackrx")

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

# --- Authentication (Unchanged) ---
auth_scheme = HTTPBearer()
EXPECTED_TOKEN = "7bf4409966a1479a8578f3258eba4e215cef0f7ccd694a2440149c1eeb4874ef"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- Main Endpoint ---
@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    """
    Processes a document and answers questions using parallel Agno agents.
    """
    start = time.perf_counter()

    manager = ml_models.get("embedding_manager")
    if not manager:
        raise HTTPException(status_code=503, detail="Embedding manager is not ready.")

    embedding_model = manager.model

    try:
        t0 = time.perf_counter()
        print(f"Processing document from: {request.documents}")
        document_bytes = await download_document(request.documents)
        document_text = await process_document(request.documents, document_bytes)

        chunks = await optimized_semantic_chunk_text(
            document_text, 
            embedding_model,
            max_chunk_size=1200, 
        )
        t1 = time.perf_counter()
        print(f"Document processed and chunked in {t1 - t0:.2f} seconds.")

        t2 = time.perf_counter()
        knowledge_base = RequestKnowledgeBase(embedding_model)
        knowledge_base.build(chunks)
        t3 = time.perf_counter()
        print(f"Embedding & FAISS Indexing took {t3 - t2:.2f} seconds")
        
        t4 = time.perf_counter()
        tasks = [
            answer_question_with_agent(question, knowledge_base)
            for question in request.questions
        ]
        
        print(f"Spawning {len(request.questions)} Agno agents in parallel...")
        answers = await asyncio.gather(*tasks)
        print("✅ All agent tasks completed.")
        t5 = time.perf_counter()
        print(f"⏱️ Parallel Agent Execution took: {t5 - t4:.4f} seconds.")

        print("Logging Q&A pairs to qa_log.log...")
        for question, answer in zip(request.questions, answers):
            # --- THIS IS THE FIX ---
            # 1. Clean the answer string first and store it in a variable.
            cleaned_answer = answer.replace('\n', ' ')
            # 2. Use the clean variable in the f-string.
            qa_logger.info(f"Q: {question} | A: {cleaned_answer}")
        
        end = time.perf_counter()
        print(f"Total processing time: {end - start:.2f} seconds")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        qa_logger.error(f"An internal error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")