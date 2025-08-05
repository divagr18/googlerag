# api/core/document_processor.py

import io
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Tuple, Dict
from urllib.parse import urlparse
import numpy as np
import fitz  # PyMuPDF
import httpx
from .embedding_manager import OptimizedEmbeddingManager
import docx
import pandas as pd
# --- NEW: Imports for Image OCR ---
from PIL import Image
import pytesseract
from httpx import AsyncHTTPTransport

# Thread pool for CPU-bound tasks like file parsing and OCR
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- Optional: If Tesseract is not in your system's PATH, uncomment and set the path here ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


async def stream_document(url: str) -> AsyncIterator[bytes]:
    transport = AsyncHTTPTransport(retries=3)
    limits = httpx.Limits(max_connections=4)
    async with httpx.AsyncClient(limits=limits, transport=transport) as client:
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RequestError as e:
            raise ValueError(f"Error streaming document from {url} after retries: {e}")

def _extract_page_text_from_bytes(args: Tuple[bytes, int]) -> Tuple[str, int]:
    pdf_bytes, page_no = args
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.load_page(page_no).get_text(), page_no + 1
    except Exception as e:
        print(f"Error extracting page {page_no}: {e}")
        return "", page_no + 1

async def _extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> List[Tuple[str, int]]:
    pdf_bytes = pdf_stream.getvalue()
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
    except Exception as e:
        raise ValueError(f"Failed to open PDF from memory stream: {e}")
    if page_count == 0:
        return []
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(cpu_executor, _extract_page_text_from_bytes, (pdf_bytes, i))
        for i in range(page_count)
    ]
    pages_data = await asyncio.gather(*tasks)
    return pages_data

def _extract_text_from_docx_stream(docx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        document = docx.Document(docx_stream)
        full_text = "\n".join([para.text for para in document.paragraphs])
        return [(full_text, 1)]
    except Exception as e:
        raise ValueError(f"Failed to process DOCX file: {e}")

def _extract_text_from_xlsx_stream(xlsx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        sheets = pd.read_excel(xlsx_stream, sheet_name=None)
        full_text = []
        for sheet_name, df in sheets.items():
            sheet_header = f"Sheet: {sheet_name}\n"
            sheet_content = df.to_csv(index=False)
            full_text.append(sheet_header + sheet_content)
        return [("\n\n".join(full_text), 1)]
    except Exception as e:
        raise ValueError(f"Failed to process XLSX file: {e}")

# --- NEW: Helper function to extract text from an image stream using OCR ---
def _extract_text_from_image_stream(image_stream: io.BytesIO) -> List[Tuple[str, int]]:
    """
    Extracts text from an image file stream using Tesseract OCR.
    All text is assigned to page 1.
    """
    try:
        # Open the image from the byte stream
        image = Image.open(image_stream)
        # Use pytesseract to perform OCR
        text = pytesseract.image_to_string(image)
        # Return the extracted text as a single "page"
        return [(text, 1)]
    except Exception as e:
        raise ValueError(f"Failed to process image file with OCR: {e}")

async def process_document_stream(
    url: str,
    document_iterator: AsyncIterator[bytes]
) -> List[Tuple[str, int]]:
    """
    Processes a document from a stream, returning a list of (text, page_number) tuples.
    Handles PDF, DOCX, XLSX, images (PNG, JPG, etc.), and plain text. Rejects other types.
    """
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower() or ".txt"

    document_bytes_io = io.BytesIO()
    async for chunk in document_iterator:
        document_bytes_io.write(chunk)
    document_bytes_io.seek(0)

    loop = asyncio.get_running_loop()
    
    # --- UPDATED: Added logic for image files ---
    image_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif']

    if file_type == '.pdf':
        return await _extract_text_from_pdf_stream(document_bytes_io)
    elif file_type == '.docx':
        return await loop.run_in_executor(
            cpu_executor, _extract_text_from_docx_stream, document_bytes_io
        )
    elif file_type == '.xlsx':
        return await loop.run_in_executor(
            cpu_executor, _extract_text_from_xlsx_stream, document_bytes_io
        )
    elif file_type in image_formats:
        return await loop.run_in_executor(
            cpu_executor, _extract_text_from_image_stream, document_bytes_io
        )
    elif file_type in ['.txt', '.md', '.csv']:
        text = document_bytes_io.read().decode('utf-8', errors='ignore')
        return [(text, 1)]
    else:
        raise ValueError(f"Unsupported file type: '{file_type}'. Please provide a supported document or image file.")


async def optimized_semantic_chunk_text(
    pages_data: List[Tuple[str, int]],
    embedding_manager: OptimizedEmbeddingManager,
    similarity_threshold: float = 0.8,  # Increased for better semantic coherence
    target_chunk_size: int = 600,       # Reduced from 1500 for better focus
    overlap_tokens: int = 150,          # Overlap between chunks
    min_chunk_size: int = 500,          # Minimum viable chunk size
) -> Tuple[List[Dict], np.ndarray]:
    if not pages_data:
        return [], np.array([])
    
    all_paragraphs = []
    for page_text, page_num in pages_data:
        all_paragraphs.extend(smart_paragraph_split(page_text, page_num))
    
    if not all_paragraphs:
        return [], np.array([])
    
    print(f"Starting improved semantic chunking on {len(all_paragraphs)} paragraphs...")
    paragraph_texts = [p[0] for p in all_paragraphs]
    
    # Batch embedding generation
    all_embeddings: List[np.ndarray] = []
    batch_size = 256
    for i in range(0, len(paragraph_texts), batch_size):
        batch = paragraph_texts[i:i + batch_size]
        emb = embedding_manager.encode_batch(batch)
        all_embeddings.append(emb)
        await asyncio.sleep(0)
    
    para_embs = np.vstack(all_embeddings)
    
    # Calculate cosine similarities between consecutive paragraphs
    sim = np.einsum('ij,ij->i', para_embs[:-1], para_embs[1:])
    
    chunks: List[Dict] = []
    chunk_embs: List[np.ndarray] = []
    
    # Enhanced chunking algorithm with overlap and better size management
    i = 0
    while i < len(all_paragraphs):
        current_chunk_texts = []
        current_chunk_indices = []
        current_page_num = all_paragraphs[i][1]
        current_size = 0
        
        # Build chunk starting from current position
        j = i
        while j < len(all_paragraphs):
            paragraph_text, page_num = all_paragraphs[j]
            paragraph_size = len(paragraph_text)
            
            # Check if we should add this paragraph to current chunk
            should_add = True
            
            # Size constraint
            if current_size + paragraph_size > target_chunk_size and current_chunk_texts:
                should_add = False
            
            # Page boundary constraint (optional - can be relaxed for better chunks)
            if page_num != current_page_num and current_chunk_texts:
                # Allow cross-page chunks if similarity is very high
                if j > 0 and j-1 < len(sim) and sim[j-1] < similarity_threshold:
                    should_add = False
            
            # Semantic similarity constraint
            if j > i and j-1 < len(sim) and sim[j-1] < similarity_threshold and current_chunk_texts:
                # If we have enough content, break the chunk
                if current_size >= min_chunk_size:
                    should_add = False
            
            if not should_add:
                break
                
            current_chunk_texts.append(paragraph_text)
            current_chunk_indices.append(j)
            current_size += paragraph_size
            current_page_num = page_num  # Update to latest page
            j += 1
        
        # Create chunk if we have content
        if current_chunk_texts:
            chunk_text = "\n\n".join(current_chunk_texts)
            avg_emb = np.mean(para_embs[current_chunk_indices], axis=0)
            
            # Enhanced metadata
            start_page = all_paragraphs[current_chunk_indices[0]][1]
            end_page = all_paragraphs[current_chunk_indices[-1]][1]
            
            chunk_metadata = {
                "page": start_page,
                "end_page": end_page if end_page != start_page else start_page,
                "paragraph_count": len(current_chunk_texts),
                "char_count": len(chunk_text),
                "chunk_id": len(chunks)
            }
            
            chunks.append({
                "text": chunk_text, 
                "metadata": chunk_metadata
            })
            chunk_embs.append(avg_emb)
        
        # Calculate next starting position with overlap
        if current_chunk_indices:
            # Find overlap starting point
            overlap_chars = 0
            overlap_start_idx = len(current_chunk_indices) - 1
            
            # Work backwards to find overlap boundary
            for k in range(len(current_chunk_indices) - 1, -1, -1):
                para_idx = current_chunk_indices[k]
                para_text = all_paragraphs[para_idx][0]
                
                if overlap_chars + len(para_text) <= overlap_tokens:
                    overlap_chars += len(para_text)
                    overlap_start_idx = k
                else:
                    break
            
            # Move to the overlap position, but ensure we make progress
            next_i = current_chunk_indices[overlap_start_idx]
            if next_i <= i:  # Ensure we always make progress
                next_i = i + max(1, len(current_chunk_indices) // 2)
            
            i = min(next_i, len(all_paragraphs))
        else:
            i += 1  # Fallback: move to next paragraph
    
    if not chunks:
        print("Warning: No chunks were created. Using fallback chunking...")
        # Fallback: create simple chunks without semantic analysis
        return await _fallback_chunking(all_paragraphs, embedding_manager, target_chunk_size)
    
    print(f"Successfully created {len(chunks)} semantic chunks with overlap and pre-computed embeddings.")
    
    # Log chunk statistics for debugging
    chunk_sizes = [len(chunk['text']) for chunk in chunks]
    print(f"Chunk size stats - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)//len(chunk_sizes)}")
    
    return chunks, np.vstack(chunk_embs)


async def _fallback_chunking(
    all_paragraphs: List[Tuple[str, int]], 
    embedding_manager: OptimizedEmbeddingManager,
    target_chunk_size: int
) -> Tuple[List[Dict], np.ndarray]:
    """Fallback chunking when semantic chunking fails"""
    
    chunks = []
    chunk_embs = []
    current_chunk = ""
    current_page = all_paragraphs[0][1] if all_paragraphs else 1
    
    for para_text, page_num in all_paragraphs:
        if len(current_chunk) + len(para_text) > target_chunk_size and current_chunk:
            # Create chunk
            emb = embedding_manager.encode_batch([current_chunk])
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {"page": current_page, "chunk_id": len(chunks)}
            })
            chunk_embs.append(emb[0])
            
            # Start new chunk with overlap (last 100 chars)
            overlap = current_chunk[-100:] if len(current_chunk) > 100 else ""
            current_chunk = overlap + "\n\n" + para_text
            current_page = page_num
        else:
            if current_chunk:
                current_chunk += "\n\n" + para_text
            else:
                current_chunk = para_text
                current_page = page_num
    
    # Add final chunk
    if current_chunk.strip():
        emb = embedding_manager.encode_batch([current_chunk.strip()])
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {"page": current_page, "chunk_id": len(chunks)}
        })
        chunk_embs.append(emb[0])
    
    return chunks, np.vstack(chunk_embs) if chunk_embs else np.array([])


def smart_paragraph_split(text: str, page_num: int, min_paragraph_length: int = 50) -> List[Tuple[str, int]]:
    """Enhanced paragraph splitting with better heuristics"""
    
    # First try double newlines (traditional paragraphs)
    paragraphs = text.split('\n\n')
    
    result: List[Tuple[str, int]] = []
    
    for para in paragraphs:
        para = para.strip()
        
        # Skip very short paragraphs
        if len(para) < min_paragraph_length:
            continue
            
        # If paragraph is very long, try to split on single newlines
        if len(para) > 2000:
            sub_paragraphs = para.split('\n')
            current_sub_chunk = ""
            
            for sub_para in sub_paragraphs:
                sub_para = sub_para.strip()
                if not sub_para:
                    continue
                    
                if len(current_sub_chunk) + len(sub_para) > 1000 and current_sub_chunk:
                    if len(current_sub_chunk) >= min_paragraph_length:
                        result.append((current_sub_chunk, page_num))
                    current_sub_chunk = sub_para
                else:
                    if current_sub_chunk:
                        current_sub_chunk += "\n" + sub_para
                    else:
                        current_sub_chunk = sub_para
            
            # Add remaining sub-chunk
            if current_sub_chunk and len(current_sub_chunk) >= min_paragraph_length:
                result.append((current_sub_chunk, page_num))
        else:
            result.append((para, page_num))
    
    return result