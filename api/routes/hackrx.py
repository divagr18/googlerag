# api/routes/hackrx.py
import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List

# Import our core logic and the ml_models state from main
from api.state import ml_models # <-- Import from the neutral state file
from api.core.document_processor import download_document, optimized_semantic_chunk_text, process_document
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_with_agent # <-- Updated import
import time
# --- Router and Pydantic Models (Unchanged) ---
hackrx_router = APIRouter(prefix="/hackrx")
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    embedding_model = ml_models.get("embedding_model")
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model is not ready.")

    try:
        t0 = time.perf_counter()
        # Phase 1: Build the Knowledge Base (once per request)
        print(f"Processing document from: {request.documents}")
        document_bytes = await download_document(request.documents)
        # Offload CPU-bound operations
        document_text = await process_document(request.documents, document_bytes)

        # Use semantic chunking instead of simple chunking
        chunks = await optimized_semantic_chunk_text(
            document_text, 
            embedding_model,
            min_chunk_size=800,
            max_chunk_size=1200, 
            similarity_threshold=0.3
        )
        t1 = time.perf_counter()
        print(f"Document processed and chunked in {t1 - t0:.2f} seconds.")

        t2 = time.perf_counter()
        knowledge_base = RequestKnowledgeBase(embedding_model)
        knowledge_base.build(chunks)
        t3 = time.perf_counter()
        print(f"Embedding & FAISS Indexing took {t3 - t2:.2f} seconds")
        
        # Phase 2: Run Parallel Agent Tasks
        # Create a list of concurrent tasks, one for each question.
        # Each task will run our agent logic with the shared knowledge base.
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

        end = time.perf_counter()
        print(f"Total processing time: {end - start:.2f} seconds")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")