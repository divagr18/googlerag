# api/routes/hackrx.py
import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List

# Import our core logic and the ml_models state from main
from api.state import ml_models # <-- Import from the neutral state file
from api.core.document_processor import download_document, extract_text_from_pdf_bytes, chunk_text
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_with_agent # <-- Updated import
import time
# --- Router and Pydantic Models (Unchanged) ---
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

    embedding_model = ml_models.get("embedding_model")
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model is not ready.")

    try:
        # Phase 1: Build the Knowledge Base (once per request)
        print(f"Processing document from: {request.documents}")
        pdf_bytes = await download_document(request.documents)
        document_text = extract_text_from_pdf_bytes(pdf_bytes)
        chunks = chunk_text(document_text)
        
        knowledge_base = RequestKnowledgeBase(embedding_model)
        knowledge_base.build(chunks)
        
        # Phase 2: Run Parallel Agent Tasks
        # Create a list of concurrent tasks, one for each question.
        # Each task will run our agent logic with the shared knowledge base.
        tasks = [
            answer_question_with_agent(question, knowledge_base)
            for question in request.questions
        ]
        
        print(f"Spawning {len(request.questions)} Agno agents in parallel...")
        answers = await asyncio.gather(*tasks)
        print("✅ All agent tasks completed.")
        end = time.perf_counter()
        print(f"Total processing time: {end - start:.2f} seconds")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")