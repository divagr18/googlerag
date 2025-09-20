import os
import asyncio
from fastapi import APIRouter, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
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
if not qa_logger.handlers:
    qa_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("qa_log.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - Q: %(message)s")
    file_handler.setFormatter(formatter)
    qa_logger.addHandler(file_handler)
    qa_logger.propagate = False

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
    answer_questions_batch_orchestrator,
    synthesize_direct_answer,
)
from api.core.embedding_manager import OptimizedEmbeddingManager

# --- New Agno Agent Import ---
from api.core.agno_agent import (
    process_with_agno_agent_simple,
    should_use_direct_processing,
)

# --- API Router and Pydantic Models ---
hackrx_router = APIRouter(prefix="/hackrx")


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


class AskRequest(BaseModel):
    questions: List[str] = Field(
        ..., description="A list of questions to answer using the stored documents."
    )
    document_ids: List[str] = Field(
        default=None, description="Optional list of specific document IDs to search within."
    )


class AskResponse(BaseModel):
    answers: List[str]


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


# --- /upload Endpoint ---
@hackrx_router.post("/upload", response_model=UploadResponse)
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
        
        processing_time = time.perf_counter() - start_time
        print(f"✅ Document uploaded successfully in {processing_time:.2f}s")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_id,
            processing_time=processing_time,
            total_chunks=len(chunks)
        )
        
    except HTTPException:
        raise
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


# --- /upload-file Endpoint ---
@hackrx_router.post("/upload-file", response_model=UploadResponse)
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
        
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Use the temporary file path as the document URL
            document_url = f"file://{temp_file_path}"
            
            # Check if document with this filename already exists
            # We'll use the filename as a unique identifier
            filename_url = f"uploaded://{filename}"
            if chroma_manager.document_exists(filename_url):
                raise HTTPException(
                    status_code=409, 
                    detail=f"File '{filename}' already exists in database. Use a different filename or delete the existing file first."
                )
            
            # Process document directly from local file (no downloading needed)
            print(f"🔄 Processing uploaded file: {filename}")
            
            # For local files, we can process them directly without streaming/downloading
            # Just read the file and pass it to the document processor
            pages_data = await process_local_file(temp_file_path)
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
            
            processing_time = time.perf_counter() - start_time
            print(f"✅ File uploaded successfully in {processing_time:.2f}s")
            
            return UploadResponse(
                message=f"File '{filename}' uploaded and processed successfully",
                document_id=document_id,
                processing_time=processing_time,
                total_chunks=len(chunks)
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File might already be deleted
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# --- /ask Endpoint ---
@hackrx_router.post("/ask", response_model=AskResponse)
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


# --- /run Endpoint with restored RAG logic ---
@hackrx_router.post(
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

@hackrx_router.get("/documents")
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


@hackrx_router.get("/documents/{document_id}")
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


@hackrx_router.delete("/documents/{document_id}")
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


@hackrx_router.post("/documents/reprocess")
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


@hackrx_router.get("/stats")
async def get_database_stats():
    """Get ChromaDB database statistics."""
    try:
        chroma_manager = ChromaDocumentManager()
        stats = chroma_manager.get_stats()
        return JSONResponse(content={"stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@hackrx_router.get("/file/{document_id}")
async def serve_file(document_id: str):
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
            else:
                content_type = "application/octet-stream"
            
            from fastapi.responses import FileResponse
            return FileResponse(
                path=file_path,
                media_type=content_type,
                filename=filename
            )
        else:
            raise HTTPException(status_code=400, detail="File is not an uploaded document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")
