import io
import os
import re
import fitz  # PyMuPDF
import docx  # python-docx
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    # Decode with utf-8, ignoring errors for robustness
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
    # 1. Determine file type from URL
    try:
        path = urlparse(url).path
        file_type = os.path.splitext(path)[1]
        if not file_type:
            raise ValueError("Could not determine file type from URL.")
    except Exception as e:
        raise ValueError(f"Invalid document URL provided: {e}")

    # 2. Run the synchronous, CPU-bound extraction in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        pdf_executor, 
        extract_text_from_document_bytes, 
        document_bytes, 
        file_type
    )

# --- Semantic Chunking Functions ---

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    # Improved sentence splitting pattern that handles various cases
    sentence_endings = r'[.!?]+(?:\s*["\'\)\]]*)?'
    sentences = []
    
    # Split by paragraphs first to maintain structure
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Split paragraph into sentences
        para_sentences = re.split(f'({sentence_endings})\\s+(?=[A-Z])', paragraph)
        
        current_sentence = ""
        for i, part in enumerate(para_sentences):
            if re.match(sentence_endings, part):
                current_sentence += part
                if current_sentence.strip() and len(current_sentence.strip()) > 10:
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            elif part.strip():
                current_sentence += part
        
        # Add any remaining sentence
        if current_sentence.strip() and len(current_sentence.strip()) > 10:
            sentences.append(current_sentence.strip())
    
    return sentences

def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity matrix between sentence embeddings."""
    return cosine_similarity(embeddings)

def find_semantic_breakpoints(similarity_matrix: np.ndarray, threshold: float = 0.3) -> List[int]:
    """
    Find breakpoints where semantic similarity drops significantly.
    Returns indices where new chunks should start.
    """
    breakpoints = [0]  # Always start with the first sentence
    
    for i in range(1, len(similarity_matrix)):
        # Calculate average similarity with previous sentences in current chunk
        chunk_start = breakpoints[-1]
        
        if i > chunk_start:
            # Average similarity with sentences in current chunk
            chunk_similarities = similarity_matrix[i, chunk_start:i]
            avg_similarity = np.mean(chunk_similarities)
            
            # If average similarity drops below threshold, create a breakpoint
            if avg_similarity < threshold:
                breakpoints.append(i)
    
    return breakpoints

async def optimized_semantic_chunk_text(text: str, embedding_model, 
                                      min_chunk_size: int = 800,      # Increased
                                      max_chunk_size: int = 1200,     # Increased  
                                      similarity_threshold: float = 0.25) -> List[str]:  # Slightly lower threshold
    """
    Optimized semantic chunking for faster processing with fewer, larger chunks.
    """
    if not text.strip():
        return []
    
    print("Starting optimized semantic chunking...")
    
    # Step 1: More aggressive sentence splitting to create larger initial units
    sentences = split_into_sentences_optimized(text)

    
    print(f"Split into {len(sentences)} sentences")
    
    # Step 2: Pre-filter very short sentences to reduce embedding calls
    filtered_sentences = []
    current_batch = ""
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:  # Very short sentences get merged
            current_batch += " " + sentence.strip()
        else:
            if current_batch:
                filtered_sentences.append(current_batch.strip())
                current_batch = ""
            filtered_sentences.append(sentence.strip())
    
    if current_batch:
        filtered_sentences.append(current_batch.strip())
    
    sentences = filtered_sentences
    print(f"Filtered to {len(sentences)} meaningful sentences")
    
    # Step 3: Generate embeddings with progress tracking
    print("Generating sentence embeddings...")
    embeddings = embedding_model.encode(
        sentences, 
        show_progress_bar=False,
        batch_size=64,  # Smaller batch size for sentence embeddings
        convert_to_numpy=True
    )
    
    # Step 4: More efficient similarity calculation
    similarity_matrix = calculate_similarity_matrix_fast(embeddings)
    
    # Step 5: Find breakpoints with adjusted threshold
    breakpoints = find_semantic_breakpoints_optimized(similarity_matrix, similarity_threshold)
    breakpoints.append(len(sentences))
    
    print(f"Found {len(breakpoints)-1} semantic segments")
    
    # Step 6: Create larger chunks with better merging logic
    chunks = create_optimized_chunks(sentences, breakpoints, min_chunk_size, max_chunk_size)
    
    # Add prefix and clean up
    final_chunks = [f"search_document: {chunk.strip()}" for chunk in chunks if chunk.strip()]
    
    print(f"Successfully created {len(final_chunks)} optimized semantic chunks")
    return final_chunks

def split_into_sentences_optimized(text: str) -> List[str]:
    """Optimized sentence splitting with better performance."""
    import re
    
    # More efficient regex pattern
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z\d])'
    
    # Split by paragraphs first to maintain structure
    paragraphs = text.split('\n\n')
    sentences = []
    
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 30:  # Skip very short paragraphs
            continue
            
        # Split paragraph into sentences
        para_sentences = re.split(sentence_pattern, paragraph.strip())
        
        # Filter and clean sentences
        for sentence in para_sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 15:  # Only keep meaningful sentences
                sentences.append(clean_sentence)
    
    return sentences

def calculate_similarity_matrix_fast(embeddings: np.ndarray) -> np.ndarray:
    """Faster similarity matrix calculation."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Only calculate upper triangle to save computation
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    # Calculate similarity in chunks to manage memory
    chunk_size = 100
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            
            # Calculate similarity for this chunk
            sim_chunk = cosine_similarity(embeddings[i:end_i], embeddings[j:end_j])
            similarity_matrix[i:end_i, j:end_j] = sim_chunk
            
            # Fill symmetric part
            if i != j:
                similarity_matrix[j:end_j, i:end_i] = sim_chunk.T
    
    return similarity_matrix

def find_semantic_breakpoints_optimized(similarity_matrix: np.ndarray, threshold: float = 0.25) -> List[int]:
    """Optimized breakpoint detection for larger chunks."""
    breakpoints = [0]
    window_size = 5  # Look at similarity with last 5 sentences
    
    for i in range(window_size, len(similarity_matrix)):
        # Calculate average similarity with recent sentences
        recent_similarities = similarity_matrix[i, max(0, i-window_size):i]
        avg_similarity = np.mean(recent_similarities)
        
        # More conservative breakpoint detection for larger chunks
        if avg_similarity < threshold:
            breakpoints.append(i)
    
    return breakpoints

def create_optimized_chunks(sentences: List[str], breakpoints: List[int], 
                          min_size: int, max_size: int) -> List[str]:
    """Create optimized chunks with better size management."""
    chunks = []
    
    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]
        
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(chunk_sentences)
        
        # If chunk is too small, try to merge with next
        if len(chunk_text) < min_size and i < len(breakpoints) - 2:
            next_start = breakpoints[i + 1]
            next_end = breakpoints[i + 2]
            next_sentences = sentences[next_start:next_end]
            combined_text = ' '.join(chunk_sentences + next_sentences)
            
            if len(combined_text) <= max_size:
                chunks.append(combined_text)
                continue  # Skip next iteration
        
        # If chunk is too large, split it
        if len(chunk_text) > max_size:
            sub_chunks = split_large_chunk_optimized(chunk_sentences, max_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)
    
    return chunks

def split_large_chunk_optimized(sentences: List[str], max_size: int) -> List[str]:
    """Optimized large chunk splitting."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# --- Legacy chunking function (kept for backward compatibility) ---