import os
import asyncio
import json
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
import time
import logging
from urllib.parse import urlparse
import httpx
from langdetect import detect
import tempfile
import shutil
import numpy as np

# --- Logger Configuration ---
qa_logger = logging.getLogger("qa_logger")
logger = logging.getLogger("guardian_score_logger")
if not qa_logger.handlers:
    qa_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("qa_log.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - Q: %(message)s")
    file_handler.setFormatter(formatter)
    qa_logger.addHandler(file_handler)
    qa_logger.propagate = False

if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - Guardian: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

# --- Core Logic Imports ---
from api.state import ml_models
from api.core.document_processor import (
    stream_document,
    process_document_stream,
    optimized_semantic_chunk_text,
    _extract_text_from_pdf_stream,
    _extract_text_from_docx_stream,
)
from api.core.vector_store import RequestKnowledgeBase
from api.core.chroma_manager import ChromaDocumentManager
from api.core.citation_utils import CitationManager
from api.core.agent_logic import (
    answer_raw_text_query,
    prepare_query_strategies_for_all_questions,
    answer_image_query,
    answer_image_query_streaming,
    answer_questions_batch_orchestrator,
    synthesize_direct_answer,
)
from api.core.embedding_manager import OptimizedEmbeddingManager

# --- New Agno Agent Import ---
from api.core.agno_agent import (
    process_with_agno_agent_simple,
    should_use_direct_processing,
)

# --- Ideal Contract System Import ---
from api.core.ideal_contract_manager import (
    IdealContractManager,
    ContractCategory,
    RiskLevel,
    create_sample_rental_template,
    create_sample_employment_template
)
from api.core.guardian_score import (
    GuardianScoreAnalyzer, 
    GuardianScoreResult, 
    ExploitationFlag,
    RiskLevel as GuardianRiskLevel
)
from api.core.document_classifier import DocumentClassifier
from api.core.ai_recommendation_engine import AIRecommendationEngine


def extract_category_from_filename(filename: str):
    """
    Extract category from filename, handling multi-word categories properly.
    Tries to match against valid ContractCategory values.
    """
    # Remove .pdf extension
    name_without_ext = filename.replace('.pdf', '')
    parts = name_without_ext.split('_')
    
    # Get all valid category values
    valid_categories = [cat.value for cat in ContractCategory]
    
    # Try progressively longer category combinations
    for i in range(1, len(parts) + 1):
        potential_category = '_'.join(parts[:i])
        if potential_category in valid_categories:
            description_parts = parts[i:]
            if description_parts:  # Ensure there's still a description part
                description = '_'.join(description_parts).replace('_', ' ').title()
                return potential_category, description
    
    return None, None


# --- API Router and Pydantic Models ---
ragsys_router = APIRouter(prefix="/ragsys")


class RunRequest(BaseModel):
    documents: str = Field(
        ..., description="URL of the PDF, DOCX, image, or text document to process."
    )
    questions: List[str] = Field(
        ..., description="A list of questions to answer based on the document."
    )


class RunResponse(BaseModel):
    answers: List[str]


class UploadRequest(BaseModel):
    document_url: str = Field(
        ..., description="URL of the document to upload and store in the vector database."
    )


class UploadResponse(BaseModel):
    message: str
    document_id: str
    processing_time: float
    total_chunks: int
    # Contract analysis results
    is_contract: bool = False
    contract_type: Optional[str] = None
    classification_confidence: float = 0.0
    guardian_score: Optional[int] = None
    risk_level: Optional[str] = None
    exploitation_flags: Optional[List[Dict]] = None
    analysis_summary: Optional[str] = None


class AskRequest(BaseModel):
    questions: List[str] = Field(
        ..., description="A list of questions to answer using the stored documents."
    )
    document_ids: List[str] = Field(
        default=None, description="Optional list of specific document IDs to search within."
    )


class AskResponse(BaseModel):
    answers: List[str]


# --- Ideal Contract Models ---
class IdealContractClause(BaseModel):
    name: str
    description: str
    importance: int = Field(..., ge=1, le=10, description="Importance score from 1-10")
    keywords: List[str]
    required: bool


class IdealContractRiskFactor(BaseModel):
    name: str
    description: str
    risk_level: str
    keywords: List[str]
    penalty_score: int = Field(..., description="Negative score penalty for this risk")


class IdealContractComplianceRequirement(BaseModel):
    name: str
    description: str
    required: bool
    keywords: List[str]


class CreateIdealContractRequest(BaseModel):
    category: str = Field(..., description="Contract category (rental, employment, etc.)")
    title: str
    description: str
    essential_clauses: List[IdealContractClause]
    risk_factors: List[IdealContractRiskFactor]
    compliance_requirements: List[IdealContractComplianceRequirement]
    scoring_weights: Dict[str, float] = Field(
        default={"essential_clauses": 0.6, "risk_factors": 0.3, "compliance": 0.1},
        description="Weights for scoring components (must sum to 1.0)"
    )


class IdealContractResponse(BaseModel):
    template_id: str
    category: str
    title: str
    description: str
    created_timestamp: str
    total_essential_clauses: int
    total_risk_factors: int
    total_compliance_requirements: int


class ListIdealContractsResponse(BaseModel):
    templates: List[IdealContractResponse]
    total: int


# --- Image Analysis Models ---
class ImageAnalysisRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the image")


class ImageAnalysisResponse(BaseModel):
    answer: str
    processing_time: float


async def process_local_file(file_path: str) -> List[Tuple[str, int]]:
    """
    Process a local file directly without downloading.
    Returns a list of (text, page_number) tuples.
    """
    import io
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Create a thread pool executor for CPU-bound tasks
    cpu_executor = ThreadPoolExecutor(max_workers=2)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == ".pdf":
            # Process PDF
            with open(file_path, 'rb') as f:
                pdf_stream = io.BytesIO(f.read())
            return await _extract_text_from_pdf_stream(pdf_stream)
            
        elif file_ext == ".docx":
            # Process DOCX
            with open(file_path, 'rb') as f:
                docx_stream = io.BytesIO(f.read())
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(cpu_executor, _extract_text_from_docx_stream, docx_stream)
            
        elif file_ext in [".txt", ".md"]:
            # Process text files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [(content, 1)]  # Single page for text files
            
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
    except Exception as e:
        print(f"Error processing local file {file_path}: {e}")
        raise


def is_image_url(url: str) -> bool:
    image_formats = [".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".gif", ".webp"]
    path = urlparse(url).path
    file_ext = os.path.splitext(path)[1].lower()
    return file_ext in image_formats


class UnsupportedFileType(Exception):
    pass


def validate_file_type(url: str) -> None:
    # Check if it's a local file path instead of URL
    if os.path.isabs(url) or url.startswith('C:') or url.startswith('/'):
        raise UnsupportedFileType(
            "Local file paths are not supported in /upload endpoint. "
            "Use /upload-file endpoint to upload local files."
        )
    
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

    # Initialize ChromaDB manager
    chroma_manager = ChromaDocumentManager()
    
    # Check if document already exists in persistent storage
    if chroma_manager.document_exists(document_url):
        print(f"📋 Document found in persistent storage: {document_url[:50]}...")
        
        # Create knowledge base from existing embeddings
        knowledge_base = RequestKnowledgeBase(manager)
        
        # For now, we'll still need to rebuild the FAISS index from ChromaDB
        # In a future optimization, we could cache the FAISS index as well
        print(f"⚡ Using cached embeddings, rebuilding search index...")
        
        # Get document info
        doc_info = chroma_manager.get_document_info(document_url)
        if doc_info:
            print(f"📖 Document: {doc_info['document_title']} ({doc_info['total_chunks']} chunks)")
        
        # TODO: Load chunks from ChromaDB and rebuild FAISS index
        # For now, continue with normal processing but will store in ChromaDB
        
    t0 = time.perf_counter()
    doc_iterator = stream_document(document_url)
    pages_data = await process_document_stream(document_url, doc_iterator)
    t1 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Download & text extraction took: {t1 - t0:.2f}s")

    t2 = time.perf_counter()
    chunks, precomputed_embeddings = await optimized_semantic_chunk_text(
        pages_data, manager, document_url=document_url
    )
    t3 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ Semantic chunking (incl. embedding) took: {t3 - t2:.2f}s")

    t4 = time.perf_counter()
    knowledge_base = RequestKnowledgeBase(manager)

    if chunks:
        await knowledge_base.build(chunks, precomputed_embeddings)
        
        # Store in ChromaDB for future use
        try:
            document_id = chroma_manager.store_document_chunks(
                document_url, chunks, precomputed_embeddings
            )
            print(f"💾 Stored document in persistent storage: {document_id}")
        except Exception as e:
            print(f"⚠️ Failed to store in ChromaDB: {e}")
            # Continue without ChromaDB storage
    else:
        print("PIPE-DOC: ⚠️ No chunks were generated from the document.")

    t5 = time.perf_counter()
    print(f"PIPE-DOC: ⏱️ KB indexing took: {t5 - t4:.2f}s")
    print(
        f"PIPE-DOC: ✅ Full document pipeline complete in {time.perf_counter() - pipeline_start_time:.2f}s."
    )
    return knowledge_base


# --- Helper Functions ---
async def analyze_contract_if_applicable(
    pages_data: List[Tuple[str, int]], 
    document_url: str, 
    chunks: List[str]
) -> Dict[str, any]:
    """
    Automatically classify document and run Guardian Score analysis if it's a contract
    """
    try:
        # Extract full text from pages_data
        full_text = "\n".join([page[0] for page in pages_data])
        filename = document_url.split("/")[-1] if "/" in document_url else document_url
        
        # Initialize document classifier
        classifier = DocumentClassifier()
        
        # Classify the document
        classification = classifier.classify_document(full_text, filename)
        
        print(f"📋 Document classification: {classification}")
        
        # Base response
        analysis_results = {
            "is_contract": classification["is_contract"],
            "contract_type": classification["contract_type"],
            "classification_confidence": classification["confidence"]
        }
        
        # If it's a contract and should be analyzed, run Guardian Score
        if classification["should_analyze"]:
            try:
                print(f"🛡️ Running Guardian Score analysis for {classification['contract_type']} contract")
                
                # Initialize Guardian Score analyzer (now uses Gemini directly)
                guardian_analyzer = GuardianScoreAnalyzer()
                
                # Run Guardian Score analysis
                guardian_result = await guardian_analyzer.analyze_contract(full_text)
                
                # Format exploitation flags for response
                formatted_flags = [
                    {
                        "type": flag.type.value,
                        "risk_level": flag.risk_level.value,
                        "description": flag.description,
                        "clause_text": flag.clause_text,
                        "severity_score": flag.severity_score,
                        "recommendation": flag.recommendation,
                        "ai_recommendation": flag.ai_recommendation
                    }
                    for flag in guardian_result.exploitation_flags
                ]
                
                analysis_results.update({
                    "guardian_score": guardian_result.overall_score,
                    "risk_level": guardian_result.risk_level.value,
                    "exploitation_flags": formatted_flags,
                    "analysis_summary": guardian_result.summary
                })
                
                print(f"🛡️ Guardian Score: {guardian_result.overall_score}/100 ({guardian_result.risk_level.value})")
                
            except Exception as e:
                logger.error(f"Guardian Score analysis failed: {str(e)}")
                analysis_results.update({
                    "guardian_score": None,
                    "risk_level": "unknown",
                    "exploitation_flags": [],
                    "analysis_summary": f"Analysis failed: {str(e)}"
                })
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Contract analysis failed: {str(e)}")
        return {
            "is_contract": False,
            "contract_type": None,
            "classification_confidence": 0.0,
            "guardian_score": None,
            "risk_level": None,
            "exploitation_flags": None,
            "analysis_summary": f"Classification failed: {str(e)}"
        }


# --- /upload Endpoint ---
@ragsys_router.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest = Body(...)):
    """
    Upload and process a document into the persistent vector database.
    This endpoint allows you to pre-process documents and store them for future queries.
    """
    try:
        start_time = time.perf_counter()
        document_url = request.document_url
        
        print(f"📤 Starting document upload: {document_url}")
        
        # Validate file type
        validate_file_type(document_url)
        
        # Initialize components
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(status_code=503, detail="Embedding manager not ready")
        chroma_manager = ChromaDocumentManager()
        
        # Check if document already exists
        if chroma_manager.document_exists(document_url):
            raise HTTPException(
                status_code=409, 
                detail=f"Document already exists in database. Use /documents/reprocess to update it."
            )
        
        # Process document
        print(f"🔄 Processing document: {document_url}")
        doc_iterator = stream_document(document_url)
        pages_data = await process_document_stream(document_url, doc_iterator)
        chunks, precomputed_embeddings = await optimized_semantic_chunk_text(
            pages_data, manager, document_url=document_url
        )
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No chunks could be generated from the document. Please check the document format and content."
            )
        
        # Store in ChromaDB
        document_id = chroma_manager.store_document_chunks(
            document_url, chunks, precomputed_embeddings
        )
        
        # Automatic contract classification and Guardian Score analysis
        analysis_results = await analyze_contract_if_applicable(
            pages_data, document_url, chunks
        )
        
        processing_time = time.perf_counter() - start_time
        print(f"✅ Document uploaded successfully in {processing_time:.2f}s")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_id,
            processing_time=processing_time,
            total_chunks=len(chunks),
            **analysis_results  # Include contract analysis results
        )
        
    except HTTPException:
        raise
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


# --- /upload-file Endpoint ---
@ragsys_router.post("/upload-file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    custom_filename: Optional[str] = None
):
    """
    Upload and process a local file into the persistent vector database.
    Supports PDF, DOCX, TXT, and other common document formats.
    
    Args:
        file: The file to upload
        custom_filename: Optional custom name for the file (will use original filename if not provided)
    """
    try:
        start_time = time.perf_counter()
        
        # Validate file type
        filename = custom_filename or file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        print(f"📤 Starting file upload: {filename}")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.txt', '.doc', '.md'}
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Initialize components
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(status_code=503, detail="Embedding manager not ready")
        chroma_manager = ChromaDocumentManager()
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file permanently to uploads directory
        file_path = os.path.join(uploads_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Use the filename URL for database storage
            filename_url = f"uploaded://{filename}"
            
            # Check if document with this filename already exists
            if chroma_manager.document_exists(filename_url):
                # Remove the saved file if document already exists
                os.unlink(file_path)
                raise HTTPException(
                    status_code=409, 
                    detail=f"File '{filename}' already exists in database. Use a different filename or delete the existing file first."
                )
            
            # Process document directly from local file
            print(f"🔄 Processing uploaded file: {filename}")
            
            # Process the permanently saved file
            pages_data = await process_local_file(file_path)
            chunks, precomputed_embeddings = await optimized_semantic_chunk_text(
                pages_data, manager, document_url=filename_url  # Use filename_url for storage
            )
            
            if not chunks:
                raise HTTPException(
                    status_code=400, 
                    detail="No chunks could be generated from the file. Please check the file format and content."
                )
            
            # Store in ChromaDB with the filename URL
            document_id = chroma_manager.store_document_chunks(
                filename_url, chunks, precomputed_embeddings
            )
            
            # Automatic contract classification and Guardian Score analysis
            analysis_results = await analyze_contract_if_applicable(
                pages_data, filename_url, chunks
            )
            
            # Update ChromaDB with analysis results if any Guardian Score data was generated
            if analysis_results.get("guardian_score") is not None:
                try:
                    # Update the document with analysis data
                    document_id = chroma_manager.store_document_chunks(
                        filename_url, chunks, precomputed_embeddings, 
                        force_update=True, analysis_data=analysis_results
                    )
                    print(f"🛡️ Updated document with Guardian Score analysis: {analysis_results.get('guardian_score')}/100")
                except Exception as e:
                    print(f"⚠️ Failed to update document with analysis data: {e}")
                    # Continue without failing the upload
            
            processing_time = time.perf_counter() - start_time
            print(f"✅ File uploaded successfully in {processing_time:.2f}s")
            
            return UploadResponse(
                message=f"File '{filename}' uploaded and processed successfully",
                document_id=document_id,
                processing_time=processing_time,
                total_chunks=len(chunks),
                **analysis_results  # Include contract analysis results
            )
            
        except Exception as e:
            # Clean up saved file on error
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass  # File might already be deleted
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# --- /ask Endpoint ---
@ragsys_router.post("/ask", response_model=AskResponse)
async def ask_questions(request: AskRequest = Body(...)):
    """
    Ask questions against documents stored in the persistent vector database.
    This endpoint uses the pre-processed documents and returns answers with citations.
    """
    try:
        start_time = time.perf_counter()
        print(f"❓ Processing {len(request.questions)} questions from stored documents")
        
        # Initialize components
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(status_code=503, detail="Embedding manager not ready")
        chroma_manager = ChromaDocumentManager()
        citation_manager = CitationManager()
        
        # Check if we have any documents
        stats = chroma_manager.get_stats()
        if stats.get('total_chunks', 0) == 0:
            raise HTTPException(
                status_code=404,
                detail="No documents found in database. Please upload documents first using /upload endpoint."
            )
        
        answers = []
        for question in request.questions:
            print(f"🔍 Processing question: {question}")
            
            # Generate query embedding
            query_embedding = manager.encode_batch([question])
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding[0])
            else:
                query_embedding = query_embedding[0]
            
            # Search relevant chunks in ChromaDB
            search_results = chroma_manager.search_documents(
                query_embedding=query_embedding,
                n_results=10,
                document_ids=request.document_ids  # Filter by specific documents if provided
            )
            
            if not search_results:
                answers.append("I couldn't find relevant information to answer this question in the stored documents.")
                continue
            
            # Extract chunks for processing
            relevant_chunks = [chunk_dict for chunk_dict, _ in search_results]
            
            # Build a temporary knowledge base for this query
            knowledge_base = RequestKnowledgeBase(manager)
            
            # Generate embeddings for chunks with proper numpy conversion
            chunk_texts = [chunk['text'] for chunk in relevant_chunks]
            chunk_embeddings_raw = manager.encode_batch(chunk_texts)
            if isinstance(chunk_embeddings_raw, list):
                chunk_embeddings = np.array(chunk_embeddings_raw)
            else:
                chunk_embeddings = chunk_embeddings_raw
                
            await knowledge_base.build(relevant_chunks, chunk_embeddings)
            
            # Generate answer using existing logic
            try:
                answer_results = await answer_questions_batch_orchestrator(
                    knowledge_base, 
                    [{"original_question": question, "sub_questions": [question]}], 
                    use_high_k=True
                )
                
                # Extract the answer from the results
                if answer_results and len(answer_results) > 0:
                    base_answer = answer_results[0][0]  # First tuple, first element (answer)
                else:
                    base_answer = "I couldn't generate an answer for this question."
                
                # The answer already includes citations from the orchestrator
                answers.append(base_answer)
                
            except Exception as e:
                print(f"⚠️ Error generating answer: {e}")
                answers.append("I encountered an error while generating an answer for this question.")
        
        processing_time = time.perf_counter() - start_time
        print(f"✅ Answered {len(request.questions)} questions in {processing_time:.2f}s")
        
        # Log Q&A
        for question, answer in zip(request.questions, answers):
            try:
                qa_logger.info(f"{question} | A: {answer.replace(chr(10), ' ')}")
            except Exception:
                pass
        
        return AskResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process questions: {str(e)}")


# --- /analyze-image Endpoint ---
@ragsys_router.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    question: str = Form(..., description="Question to ask about the image")
):
    """
    Analyze an uploaded image and answer questions about it using the Gemini vision model.
    Supports common image formats (JPEG, PNG, GIF, WebP).
    Returns a streaming response.
    """
    try:
        start_time = time.perf_counter()
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size (limit to 10MB)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        image_content = await image.read()
        if len(image_content) > MAX_SIZE:
            raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB")
        
        print(f"🖼️ Analyzing image: {image.filename} ({len(image_content)} bytes)")
        print(f"❓ Question: {question}")
        
        # Create a data URL for the image
        import base64
        
        # Convert image to base64 data URL
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        # Determine the MIME type for the data URL
        mime_type = image.content_type
        image_url = f"data:{mime_type};base64,{base64_image}"
        
        async def generate_response():
            """Generator function for streaming response"""
            full_answer = ""
            try:
                async for chunk in answer_image_query_streaming(image_url, question):
                    if chunk:
                        full_answer += chunk
                        # Send chunk as JSON with proper formatting
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                processing_time = time.perf_counter() - start_time
                print(f"✅ Image analysis completed in {processing_time:.2f}s")
                
                # Log the Q&A
                try:
                    qa_logger.info(f"[IMAGE] {question} | A: {full_answer.replace(chr(10), ' ')}")
                except Exception:
                    pass
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True, 'processing_time': processing_time})}\n\n"
                
            except Exception as e:
                print(f"❌ Error in streaming image analysis: {e}")
                yield f"data: {json.dumps({'error': 'Failed to analyze image'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")


# --- /run Endpoint with restored RAG logic ---
@ragsys_router.post(
    "/run", response_model=RunResponse
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


# --- ChromaDB Management Endpoints ---

@ragsys_router.get("/documents")
async def list_documents():
    """List all documents stored in ChromaDB."""
    try:
        chroma_manager = ChromaDocumentManager()
        documents = chroma_manager.list_documents()
        stats = chroma_manager.get_stats()
        
        return JSONResponse(content={
            "documents": documents,
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@ragsys_router.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get information about a specific document."""
    try:
        chroma_manager = ChromaDocumentManager()
        
        # Find document by ID in the list
        documents = chroma_manager.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return JSONResponse(content={"document": document})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document info: {str(e)}")


@ragsys_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from ChromaDB."""
    try:
        chroma_manager = ChromaDocumentManager()
        
        # Find document URL by ID
        documents = chroma_manager.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        success = chroma_manager.delete_document(document["document_url"])
        
        if success:
            return JSONResponse(content={"message": f"Document {document_id} deleted successfully"})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@ragsys_router.post("/documents/reprocess")
async def reprocess_document(document_url: str = Body(..., embed=True)):
    """Reprocess and update a document in ChromaDB."""
    try:
        manager = ml_models.get("embedding_manager")
        if not manager:
            raise HTTPException(status_code=503, detail="Embedding manager not ready")
        chroma_manager = ChromaDocumentManager()
        
        # Check if document exists
        if not chroma_manager.document_exists(document_url):
            raise HTTPException(status_code=404, detail="Document not found in database")
        
        # Process document with force update
        pipeline_start_time = time.perf_counter()
        print(f"🔄 Reprocessing document: {document_url}")
        
        doc_iterator = stream_document(document_url)
        pages_data = await process_document_stream(document_url, doc_iterator)
        chunks, precomputed_embeddings = await optimized_semantic_chunk_text(
            pages_data, manager, document_url=document_url
        )
        
        if chunks:
            document_id = chroma_manager.store_document_chunks(
                document_url, chunks, precomputed_embeddings, force_update=True
            )
            
            processing_time = time.perf_counter() - pipeline_start_time
            print(f"✅ Document reprocessed in {processing_time:.2f}s")
            
            return JSONResponse(content={
                "message": "Document reprocessed successfully",
                "document_id": document_id,
                "processing_time": processing_time,
                "total_chunks": len(chunks)
            })
        else:
            raise HTTPException(status_code=400, detail="No chunks generated from document")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")


@ragsys_router.get("/stats")
async def get_database_stats():
    """Get ChromaDB database statistics."""
    try:
        chroma_manager = ChromaDocumentManager()
        stats = chroma_manager.get_stats()
        return JSONResponse(content={"stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@ragsys_router.get("/file/{document_id}")
async def serve_file(document_id: str, format: Optional[str] = Query(None, description="Response format: 'text' for extracted text content")):
    """Serve uploaded files by document ID"""
    try:
        chroma_manager = ChromaDocumentManager()
        
        # Get document metadata to find the file path
        documents = chroma_manager.list_documents()
        target_doc = None
        for doc in documents:
            if doc["document_id"] == document_id:
                target_doc = doc
                break
        
        if not target_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Extract filename from document_url (e.g., "uploaded://filename.pdf" -> "filename.pdf")
        if target_doc["document_url"].startswith("uploaded://"):
            filename = target_doc["document_url"].replace("uploaded://", "")
            file_path = os.path.join("uploads", filename)
            
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found on disk")
            
            # Determine content type based on file extension
            if filename.lower().endswith('.pdf'):
                content_type = "application/pdf"
            elif filename.lower().endswith('.docx'):
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif filename.lower().endswith('.doc'):
                content_type = "application/msword"
            elif filename.lower().endswith('.txt'):
                content_type = "text/plain"
            else:
                content_type = "application/octet-stream"
            
            from fastapi.responses import FileResponse
            
            # For PDFs, check if text format is requested
            if filename.lower().endswith('.pdf'):
                if format == "text":
                    # Extract text from PDF
                    try:
                        from api.core.document_processor import _extract_text_from_pdf_stream
                        from fastapi.responses import Response
                        import io
                        
                        with open(file_path, "rb") as f:
                            pdf_stream = io.BytesIO(f.read())
                            # _extract_text_from_pdf_stream returns List[Tuple[str, int]]
                            text_chunks = await _extract_text_from_pdf_stream(pdf_stream)
                            # Combine all chunks into a single text string
                            extracted_text = "\n\n".join([f"Page {chunk[1]}:\n{chunk[0]}" for chunk in text_chunks])
                            return Response(
                                content=extracted_text,
                                media_type="text/plain; charset=utf-8",
                                headers={
                                    "Content-Disposition": "inline",
                                    "Content-Type": "text/plain; charset=utf-8"
                                }
                            )
                    except Exception as e:
                        # If text extraction fails, return error message
                        print(f"PDF text extraction failed for {filename}: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        return Response(
                            content=f"Error extracting text from PDF: {str(e)}",
                            media_type="text/plain; charset=utf-8",
                            headers={
                                "Content-Disposition": "inline",
                                "Content-Type": "text/plain; charset=utf-8"
                            }
                        )
                else:
                    # Return PDF binary for normal viewing
                    from fastapi.responses import Response
                    with open(file_path, "rb") as f:
                        content = f.read()
                    return Response(
                        content=content,
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": "inline",
                            "Content-Type": "application/pdf"
                        }
                    )
            # For text files, return plain text content for better frontend handling
            elif filename.lower().endswith('.txt'):
                from fastapi.responses import Response
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return Response(
                    content=content,
                    media_type="text/plain; charset=utf-8",
                    headers={
                        "Content-Disposition": "inline",
                        "Content-Type": "text/plain; charset=utf-8"
                    }
                )
            # For Word docs, try to extract text content
            elif filename.lower().endswith(('.docx', '.doc')):
                try:
                    from api.core.document_processor import _extract_text_from_docx_stream
                    from fastapi.responses import Response
                    import io
                    
                    with open(file_path, "rb") as f:
                        if filename.lower().endswith('.docx'):
                            # Create a BytesIO stream from the file content
                            docx_stream = io.BytesIO(f.read())
                            # _extract_text_from_docx_stream returns List[Tuple[str, int]]
                            text_chunks = _extract_text_from_docx_stream(docx_stream)
                            # Combine all chunks into a single text string
                            extracted_text = "\n\n".join([chunk[0] for chunk in text_chunks])
                            return Response(
                                content=extracted_text,
                                media_type="text/plain; charset=utf-8",
                                headers={
                                    "Content-Disposition": "inline",
                                    "Content-Type": "text/plain; charset=utf-8"
                                }
                            )
                except Exception as e:
                    # If text extraction fails, fall back to file download
                    print(f"Text extraction failed for {filename}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                
                # Fallback to regular file response for Word docs
                return FileResponse(
                    path=file_path,
                    media_type=content_type,
                    filename=filename
                )
            else:
                # For other file types, use regular FileResponse
                return FileResponse(
                    path=file_path,
                    media_type=content_type,
                    filename=filename
                )
        else:
            raise HTTPException(status_code=400, detail="File is not an uploaded document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


@ragsys_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the database and file system"""
    try:
        chroma_manager = ChromaDocumentManager()
        
        # Get document metadata before deletion
        documents = chroma_manager.list_documents()
        target_doc = None
        for doc in documents:
            if doc["document_id"] == document_id:
                target_doc = doc
                break
        
        if not target_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from ChromaDB
        success = chroma_manager.delete_document_by_id(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document from database")
        
        # Delete file from uploads directory if it's an uploaded file
        if target_doc["document_url"].startswith("uploaded://"):
            filename = target_doc["document_url"].replace("uploaded://", "")
            file_path = os.path.join("uploads", filename)
            
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    print(f"✅ Deleted file: {file_path}")
                except OSError as e:
                    print(f"⚠️ Warning: Could not delete file {file_path}: {e}")
                    # Don't fail the request if file deletion fails
        
        return JSONResponse(content={
            "message": f"Document '{target_doc['document_title']}' deleted successfully",
            "document_id": document_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# === IDEAL CONTRACT MANAGEMENT ENDPOINTS ===

@ragsys_router.post("/ideal-contracts", response_model=IdealContractResponse)
async def create_ideal_contract(request: CreateIdealContractRequest):
    """
    Create a new ideal contract template for Guardian Score comparison.
    
    This endpoint allows admins to upload ideal contract templates that will be used
    as benchmarks for scoring uploaded contracts. Each template includes essential
    clauses, risk factors, and compliance requirements for a specific contract type.
    """
    try:
        # Get embedding manager from state
        embedding_manager = ml_models.get("embedding_manager")
        if not embedding_manager:
            raise HTTPException(status_code=500, detail="Embedding manager not available")
        
        # Initialize ideal contract manager
        ideal_manager = IdealContractManager()
        
        # Validate category
        try:
            category = ContractCategory(request.category.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {[c.value for c in ContractCategory]}"
            )
        
        # Validate scoring weights sum to 1.0
        total_weight = sum(request.scoring_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail=f"Scoring weights must sum to 1.0, got {total_weight}"
            )
        
        # Create embedding for the template
        template_text = f"""
        {request.title}
        {request.description}
        Essential Clauses: {[clause.description for clause in request.essential_clauses]}
        Risk Factors: {[risk.description for risk in request.risk_factors]}
        Compliance: {[comp.description for comp in request.compliance_requirements]}
        """
        
        embedding = embedding_manager.get_embedding(template_text)
        
        # Convert Pydantic models to dicts
        essential_clauses = [clause.dict() for clause in request.essential_clauses]
        risk_factors = [risk.dict() for risk in request.risk_factors]
        compliance_requirements = [comp.dict() for comp in request.compliance_requirements]
        
        # Store the ideal contract
        template_id = ideal_manager.store_ideal_contract(
            category=category,
            title=request.title,
            description=request.description,
            essential_clauses=essential_clauses,
            risk_factors=risk_factors,
            compliance_requirements=compliance_requirements,
            scoring_weights=request.scoring_weights,
            embedding=embedding,
            created_by="api_user"
        )
        
        return IdealContractResponse(
            template_id=template_id,
            category=category.value,
            title=request.title,
            description=request.description,
            created_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_essential_clauses=len(request.essential_clauses),
            total_risk_factors=len(request.risk_factors),
            total_compliance_requirements=len(request.compliance_requirements)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ideal contract: {str(e)}")


@ragsys_router.post("/ideal-contracts/seed-templates")
async def seed_ideal_contract_templates():
    """
    Seed the database with sample ideal contract templates for testing.
    
    This creates sample templates for rental and employment contracts that can be
    used to test the Guardian Score functionality.
    """
    try:
        # Get embedding manager from state
        embedding_manager = ml_models.get("embedding_manager")
        if not embedding_manager:
            raise HTTPException(status_code=500, detail="Embedding manager not available")
        
        ideal_manager = IdealContractManager()
        created_templates = []
        
        # Create sample rental template
        rental_template = create_sample_rental_template()
        rental_text = f"""
        {rental_template['title']}
        {rental_template['description']}
        Essential Clauses: {[clause['description'] for clause in rental_template['essential_clauses']]}
        Risk Factors: {[risk['description'] for risk in rental_template['risk_factors']]}
        """
        rental_embedding = embedding_manager.get_embedding(rental_text)
        
        rental_id = ideal_manager.store_ideal_contract(
            category=rental_template['category'],
            title=rental_template['title'],
            description=rental_template['description'],
            essential_clauses=rental_template['essential_clauses'],
            risk_factors=rental_template['risk_factors'],
            compliance_requirements=rental_template['compliance_requirements'],
            scoring_weights=rental_template['scoring_weights'],
            embedding=rental_embedding,
            created_by="seed_script"
        )
        created_templates.append({"type": "rental", "template_id": rental_id})
        
        # Create sample employment template
        employment_template = create_sample_employment_template()
        employment_text = f"""
        {employment_template['title']}
        {employment_template['description']}
        Essential Clauses: {[clause['description'] for clause in employment_template['essential_clauses']]}
        Risk Factors: {[risk['description'] for risk in employment_template['risk_factors']]}
        """
        employment_embedding = embedding_manager.get_embedding(employment_text)
        
        employment_id = ideal_manager.store_ideal_contract(
            category=employment_template['category'],
            title=employment_template['title'],
            description=employment_template['description'],
            essential_clauses=employment_template['essential_clauses'],
            risk_factors=employment_template['risk_factors'],
            compliance_requirements=employment_template['compliance_requirements'],
            scoring_weights=employment_template['scoring_weights'],
            embedding=employment_embedding,
            created_by="seed_script"
        )
        created_templates.append({"type": "employment", "template_id": employment_id})
        
        return {
            "message": "Sample ideal contract templates created successfully",
            "created_templates": created_templates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding templates: {str(e)}")


# Old download endpoint removed - now using folder processing approach

@ragsys_router.get("/ideal-contracts/available-sources")
async def get_available_template_sources():
    """
    Get information about the template processing system.
    """
    try:
        templates_folder = os.path.join(os.getcwd(), "ideal_contract_templates")
        
        return {
            "message": "Ideal contract template processing system",
            "templates_folder": templates_folder,
            "instructions": {
                "step_1": "Place your PDF contract templates in the 'ideal_contract_templates' folder",
                "step_2": "Name files using format: {category}_{description}.pdf",
                "step_3": "Call POST /api/v1/ragsys/ideal-contracts/process-folder-templates",
                "valid_categories": [c.value for c in ContractCategory],
                "examples": [
                    "rental_mumbai_leave_license.pdf",
                    "employment_standard_indian.pdf",
                    "nda_mutual_template.pdf"
                ]
            },
            "endpoints": {
                "process_templates": "POST /api/v1/ragsys/ideal-contracts/process-folder-templates",
                "list_templates": "GET /api/v1/ragsys/ideal-contracts",
                "get_template": "GET /api/v1/ragsys/ideal-contracts/{template_id}"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting template info: {str(e)}")


@ragsys_router.post("/ideal-contracts/process-folder-templates")
async def process_folder_templates():
    """
    Process all PDF templates from the 'ideal_contract_templates' folder.
    
    Place your ideal contract PDFs in the 'ideal_contract_templates' folder, then call this endpoint.
    The system will automatically process and classify them based on filename patterns.
    
    Filename convention: {category}_{description}.pdf
    Examples:
    - rental_mumbai_leave_license.pdf
    - employment_standard_agreement.pdf  
    - nda_mutual_agreement.pdf
    """
    try:
        templates_folder = os.path.join(os.getcwd(), "ideal_contract_templates")
        
        if not os.path.exists(templates_folder):
            raise HTTPException(
                status_code=404, 
                detail=f"Templates folder not found: {templates_folder}. Please create it and add PDF files."
            )
        
        # Get all PDF files from the folder
        pdf_files = [f for f in os.listdir(templates_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return {
                "message": "No PDF files found in templates folder",
                "folder_path": templates_folder,
                "instructions": "Add PDF files with format: {category}_{description}.pdf"
            }
        
        # Get embedding manager
        embedding_manager = ml_models.get("embedding_manager")
        if not embedding_manager:
            raise HTTPException(status_code=500, detail="Embedding manager not available")
        
        from api.core.document_processor import process_document_stream
        ideal_manager = IdealContractManager()
        
        processed_templates = []
        skipped_files = []
        
        print(f"📁 Processing {len(pdf_files)} PDF templates from folder...")
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(templates_folder, pdf_file)
                print(f"📄 Processing: {pdf_file}")
                
                # Extract category from filename using the helper function
                category_result = extract_category_from_filename(pdf_file)
                if not category_result:
                    print(f"⚠️ Skipping {pdf_file} - Invalid filename format. Use: category_description.pdf")
                    skipped_files.append({
                        "file": pdf_file,
                        "reason": "Invalid filename format. Use: category_description.pdf"
                    })
                    continue
                
                category_name, description = category_result
                
                # Validate category
                try:
                    category_enum = ContractCategory(category_name)
                except ValueError:
                    print(f"⚠️ Skipping {pdf_file} - Unknown category: {category_name}")
                    skipped_files.append({
                        "file": pdf_file,
                        "reason": f"Unknown category: {category_name}. Valid: {[c.value for c in ContractCategory]}"
                    })
                    continue
                
                # Read and process the PDF
                # Process document to extract text
                document_url = f"template_folder://{pdf_file}"
                
                # Create async iterator from file content
                async def file_iterator():
                    with open(file_path, 'rb') as f:
                        chunk_size = 8192
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                
                chunks = await process_document_stream(
                    document_url,
                    file_iterator()
                )
                
                if not chunks:
                    print(f"⚠️ Skipping {pdf_file} - Could not extract text")
                    skipped_files.append({
                        "file": pdf_file,
                        "reason": "Could not extract text from PDF"
                    })
                    continue
                
                # Combine all chunks to get full document text
                full_text = "\n".join([chunk[0] for chunk in chunks])  # chunk is (text, page_number)
                
                if len(full_text.strip()) < 100:
                    print(f"⚠️ Skipping {pdf_file} - Text too short ({len(full_text)} chars)")
                    skipped_files.append({
                        "file": pdf_file,
                        "reason": f"Extracted text too short: {len(full_text)} characters"
                    })
                    continue
                
                # Create embedding for the full document
                embedding = embedding_manager.encode_batch([full_text])[0]  # Get first embedding from batch
                
                # Create template structure
                title = f"{category_name.title()} Template - {description}"
                template_data = {
                    "category": category_enum,
                    "title": title,
                    "description": f"Legal template for {category_name} contracts - {description}",
                    "essential_clauses": [
                        {
                            "name": "standard_clauses",
                            "description": f"Standard clauses found in {category_name} contracts",
                            "importance": 10,
                            "keywords": [category_name, "contract", "agreement", "terms"],
                            "required": True
                        }
                    ],
                    "risk_factors": [
                        {
                            "name": "missing_protections",
                            "description": f"Important {category_name} protections missing from user contract",
                            "risk_level": "high",
                            "keywords": ["missing", "protection", "rights"],
                            "penalty_score": -25
                        }
                    ],
                    "compliance_requirements": [
                        {
                            "name": "legal_standards",
                            "description": f"Must meet legal standards for {category_name} contracts",
                            "required": True,
                            "keywords": ["legal", "compliance", "standard"]
                        }
                    ],
                    "scoring_weights": {
                        "essential_clauses": 0.6,
                        "risk_factors": 0.3,
                        "compliance": 0.1
                    }
                }
                
                # Store the ideal contract
                template_id = ideal_manager.store_ideal_contract(
                    category=template_data["category"],
                    title=template_data["title"],
                    description=template_data["description"],
                    essential_clauses=template_data["essential_clauses"],
                    risk_factors=template_data["risk_factors"],
                    compliance_requirements=template_data["compliance_requirements"],
                    scoring_weights=template_data["scoring_weights"],
                    embedding=embedding,
                    created_by="folder_processor",
                    source_file=pdf_file
                )
                
                processed_templates.append({
                    "file": pdf_file,
                    "template_id": template_id,
                    "category": category_name,
                    "title": title,
                    "text_length": len(full_text),
                    "chunks": len(chunks)
                })
                
                print(f"✅ Processed: {pdf_file} -> {template_id}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                skipped_files.append({
                    "file": pdf_file,
                    "reason": f"Processing error: {str(e)}"
                })
                continue
        
        return {
            "message": f"Processed {len(processed_templates)} templates successfully",
            "folder_path": templates_folder,
            "processed_templates": processed_templates,
            "skipped_files": skipped_files,
            "total_files": len(pdf_files),
            "success_count": len(processed_templates),
            "skip_count": len(skipped_files),
            "instructions": {
                "filename_format": "category_description.pdf",
                "valid_categories": [c.value for c in ContractCategory],
                "examples": [
                    "rental_mumbai_leave_license.pdf",
                    "employment_standard_indian.pdf", 
                    "nda_mutual_template.pdf"
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing folder templates: {str(e)}")


@ragsys_router.post("/ideal-contracts/upload-template")
async def upload_ideal_contract_template(
    file: UploadFile = File(...),
    category: str = Body(...),
    title: str = Body(...),
    description: str = Body(...)
):
    """
    Upload a PDF file as an ideal contract template.
    
    This is a more practical approach than downloading from websites.
    Upload actual contract templates you've obtained from lawyers or legal sources.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Validate category
        try:
            category_enum = ContractCategory(category.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {[c.value for c in ContractCategory]}"
            )
        
        # Get embedding manager
        embedding_manager = ml_models.get("embedding_manager")
        if not embedding_manager:
            raise HTTPException(status_code=500, detail="Embedding manager not available")
        
        # Read file content
        file_content = await file.read()
        
        # Process the document to extract text
        from api.core.document_processor import process_document_stream
        
        # Create async iterator from file content
        async def file_content_iterator():
            chunk_size = 8192
            for i in range(0, len(file_content), chunk_size):
                yield file_content[i:i + chunk_size]
        
        document_url = f"ideal_template://{file.filename}"
        chunks = await process_document_stream(
            document_url,
            file_content_iterator()
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Combine all chunks to get full document text
        full_text = "\n".join([chunk[0] for chunk in chunks])  # chunk is (text, page_number)
        
        # Create embedding for the full document
        embedding = embedding_manager.encode_batch([full_text])[0]  # Get first embedding from batch
        
        # Initialize ideal contract manager
        ideal_manager = IdealContractManager()
        
        # Create basic template structure for the uploaded document
        # This could be enhanced with AI analysis to extract actual clauses
        template_data = {
            "category": category_enum,
            "title": title,
            "description": description,
            "essential_clauses": [
                {
                    "name": "document_content",
                    "description": f"Complete {category} contract template content",
                    "importance": 10,
                    "keywords": ["contract", "agreement", "terms", category],
                    "required": True
                }
            ],
            "risk_factors": [
                {
                    "name": "missing_standard_clauses",
                    "description": "Standard clauses missing from user's contract",
                    "risk_level": "high",
                    "keywords": ["missing", "absent", "not included"],
                    "penalty_score": -20
                }
            ],
            "compliance_requirements": [
                {
                    "name": "legal_compliance",
                    "description": "Must meet legal standards as shown in template",
                    "required": True,
                    "keywords": ["legal", "compliant", "standard"]
                }
            ],
            "scoring_weights": {
                "essential_clauses": 0.7,
                "risk_factors": 0.2,
                "compliance": 0.1
            }
        }
        
        # Store the ideal contract
        template_id = ideal_manager.store_ideal_contract(
            category=template_data["category"],
            title=template_data["title"],
            description=template_data["description"],
            essential_clauses=template_data["essential_clauses"],
            risk_factors=template_data["risk_factors"],
            compliance_requirements=template_data["compliance_requirements"],
            scoring_weights=template_data["scoring_weights"],
            embedding=embedding,
            created_by="manual_upload"
        )
        
        return {
            "message": f"Successfully uploaded ideal contract template: {title}",
            "template_id": template_id,
            "category": category,
            "title": title,
            "chunks_processed": len(chunks),
            "text_length": len(full_text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload ideal contract template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@ragsys_router.post("/ideal-contracts/analyze-contract", response_model=Dict[str, Any])
async def analyze_contract_guardian_score(
    contract_text: str = Form(...),
    category: str = Form(...),
    title: str = Form("Contract Analysis")
):
    """
    Analyze a contract using Guardian Score system to detect exploitation
    and compare against ideal templates.
    """
    try:
        # Validate category
        try:
            contract_category = ContractCategory(category.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {[c.value for c in ContractCategory]}"
            )
        
        # Initialize Guardian Score analyzer with AI recommendations
        guardian_analyzer = GuardianScoreAnalyzer()
        
        # Initialize AI recommendation engine
        try:
            from api.core.ai_recommendation_engine import AIRecommendationEngine
            ai_engine = AIRecommendationEngine()
            guardian_analyzer.set_ai_engine(ai_engine)
            logger.info("AI recommendation engine initialized")
        except Exception as e:
            logger.warning(f"AI engine initialization failed, using static recommendations: {str(e)}")
        
        # Get ideal template for comparison (optional)
        ideal_manager = IdealContractManager()
        
        # Get embedding manager to create embedding for search
        embedding_manager = ml_models.get("embedding_manager")
        ideal_template_text = None
        
        if embedding_manager:
            try:
                # Create embedding for the contract text
                contract_embedding = embedding_manager.encode_batch([contract_text])[0]
                
                # Search for similar ideal templates
                ideal_templates = ideal_manager.search_similar_templates(
                    contract_embedding, 
                    category=contract_category, 
                    n_results=1
                )
                
                if ideal_templates and len(ideal_templates) > 0:
                    # ideal_templates returns List[Tuple[Dict, float]]
                    template_data = ideal_templates[0][0]  # Get the dict from first tuple
                    ideal_template_text = template_data.get("content", "")
            except Exception as e:
                logger.warning(f"Could not retrieve ideal template: {str(e)}")
                # Continue without ideal template comparison
        
        # Analyze the contract with AI-powered recommendations
        analysis_result = await guardian_analyzer.analyze_contract(
            contract_text, 
            ideal_template_text
        )
        
        # Convert to serializable format
        result = {
            "guardian_score": analysis_result.overall_score,
            "risk_level": analysis_result.risk_level.value,
            "category": category,
            "title": title,
            "summary": analysis_result.summary,
            "exploitation_flags": [
                {
                    "type": flag.type.value,
                    "risk_level": flag.risk_level.value,
                    "description": flag.description,
                    "clause_text": flag.clause_text,
                    "severity_score": flag.severity_score,
                    "recommendation": flag.recommendation,
                    "ai_recommendation": flag.ai_recommendation
                }
                for flag in analysis_result.exploitation_flags
            ],
            "missing_protections": analysis_result.missing_protections,
            "fair_clauses": analysis_result.fair_clauses,
            "ideal_template_found": ideal_template_text is not None,
            "analysis_timestamp": "2025-09-20T12:00:00Z"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Guardian Score analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading ideal contract template: {str(e)}")
