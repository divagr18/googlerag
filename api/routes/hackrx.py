# api/routes/hackrx.py
import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import time

from api.state import ml_models
from api.core.document_processor import download_document, optimized_semantic_chunk_text, process_document
from api.core.vector_store import RequestKnowledgeBase
from api.core.agent_logic import answer_question_with_agent

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

@hackrx_router.post("/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest = Body(...)):
    """
    Processes a document and answers questions using a streamlined, optimized pipeline.
    """
    start_time = time.perf_counter()

    embedding_model = ml_models.get("embedding_model")
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model is not ready.")

    try:
        # Phase 1: Build Knowledge Base
        document_bytes = await download_document(request.documents)
        document_text = await process_document(request.documents, document_bytes)
        
        # Using the single, unified chunking strategy
        chunks = await optimized_semantic_chunk_text(document_text, embedding_model)
        
        # Using the single, unified FAISS vector store
        knowledge_base = RequestKnowledgeBase(embedding_model)
        knowledge_base.build(chunks)
        
        # Phase 2: Run Parallel Agent Tasks
        tasks = [
            answer_question_with_agent(question, knowledge_base)
            for question in request.questions
        ]
        
        answers = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        return RunResponse(answers=answers)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # It's good practice to log the actual exception
        print(f"An internal error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")