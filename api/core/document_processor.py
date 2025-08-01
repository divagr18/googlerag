import io
import os
import re
import fitz  # PyMuPDF
import docx  # python-docx
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.parse import urlparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Create a thread pool for CPU-bound parsing tasks
pdf_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def download_document(url: str) -> bytes:
    """Asynchronously downloads a document from a URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise ValueError(f"Error downloading document from {url}: {e}")

# --- Private, specific text extractors ---

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using PyMuPDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def _extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx."""
    with io.BytesIO(docx_bytes) as docx_file:
        doc = docx.Document(docx_file)
        return "\n\n".join(para.text for para in doc.paragraphs if para.text)

def _extract_text_from_txt(txt_bytes: bytes) -> str:
    """Extracts text from TXT bytes."""
    return txt_bytes.decode('utf-8', errors='ignore')

# --- Main synchronous dispatcher function ---

def extract_text_from_document_bytes(document_bytes: bytes, file_type: str) -> str:
    """
    Dispatches to the correct text extractor based on the file type.
    This function is designed to be run in a thread pool.
    """
    file_type = file_type.lower()
    
    if file_type == '.pdf':
        return _extract_text_from_pdf(document_bytes)
    elif file_type == '.docx':
        return _extract_text_from_docx(document_bytes)
    elif file_type == '.txt':
        return _extract_text_from_txt(document_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# --- Main asynchronous entry point for processing ---

async def process_document(url: str, document_bytes: bytes) -> str:
    """
    Asynchronously processes document bytes to extract text based on the URL's file type.
    """
    try:
        path = urlparse(url).path
        file_type = os.path.splitext(path)[1]
        if not file_type:
            raise ValueError("Could not determine file type from URL.")
    except Exception as e:
        raise ValueError(f"Invalid document URL provided: {e}")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        pdf_executor, 
        extract_text_from_document_bytes, 
        document_bytes, 
        file_type
    )

# --- Consolidated & Optimized Semantic Chunking ---

def preprocess_text_fast(text: str) -> str:
    """Fast text preprocessing to reduce noise"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-]', ' ', text)
    return text.strip()

def smart_paragraph_split(text: str) -> List[str]:
    """Intelligent paragraph splitting optimized for legal/financial docs"""
    paragraphs = text.split('\n\n')
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:
            if result and len(result[-1]) < 500:
                result[-1] += ' ' + para
            continue
        
        if len(para) > 2000:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current, current_len = [], 0
            for sent in sentences:
                if current_len + len(sent) > 1500 and current:
                    result.append(' '.join(current))
                    current, current_len = [sent], len(sent)
                else:
                    current.append(sent)
                    current_len += len(sent)
            if current:
                result.append(' '.join(current))
        else:
            result.append(para)
    return result

async def batch_encode_gpu_optimized(texts: List[str], model) -> np.ndarray:
    """GPU-optimized batch encoding for RTX 4060"""
    batch_size = 32 if len(texts) > 50 else 16
    
    # Generate the embeddings
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
        device='cuda' if hasattr(model, 'device') else None
    )
    
    # --- ADD THIS LINE ---
    print(f"ℹ️  [Chunking] Generated embeddings: Shape={embeddings.shape}, DType={embeddings.dtype}")
    
    return embeddings
def simple_split_chunk(text: str, max_size: int) -> List[str]:
    """Simple chunk splitting for large texts"""
    if len(text) <= max_size: return [text]
    chunks, current_chunk, current_size = [], [], 0
    for word in text.split():
        word_size = len(word) + 1
        if current_size + word_size > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_size = [word], word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    if current_chunk: chunks.append(' '.join(current_chunk))
    return chunks

def cluster_paragraphs_fast(paragraphs: List[str], embeddings: np.ndarray, target_size: int) -> List[str]:
    """Fast clustering-based chunking using MiniBatchKMeans"""
    n_paragraphs = len(paragraphs)
    if n_paragraphs <= 3: return [' '.join(paragraphs)]
    
    total_chars = sum(len(p) for p in paragraphs)
    n_clusters = max(2, min(n_paragraphs // 2, total_chars // target_size))
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)
    
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(paragraphs[i])
    
    chunks = []
    for cluster_paras in clusters.values():
        if not cluster_paras: continue
        chunk_text = '\n\n'.join(cluster_paras)
        if len(chunk_text) > target_size * 1.8:
            chunks.extend(simple_split_chunk(chunk_text, target_size))
        else:
            chunks.append(chunk_text)
    return chunks

async def optimized_semantic_chunk_text(text: str, embedding_model, max_chunk_size: int = 1200) -> List[str]:
    """
    Ultra-fast chunking optimized for performance.
    Focus: Minimize GPU memory usage, maximize throughput.
    """
    if not text.strip(): return []
    
    clean_text = preprocess_text_fast(text)
    paragraphs = smart_paragraph_split(clean_text)
    
    if len(paragraphs) <= 5:
        return [f"search_document: {chunk.strip()}" for chunk in simple_split_chunk(clean_text, max_chunk_size)]
    
    embeddings = await batch_encode_gpu_optimized(paragraphs, embedding_model)
    chunks = cluster_paragraphs_fast(paragraphs, embeddings, max_chunk_size)
    
    return [f"search_document: {chunk.strip()}" for chunk in chunks if chunk.strip()]