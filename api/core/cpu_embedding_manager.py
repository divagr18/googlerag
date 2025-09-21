import asyncio
import google.genai as genai
from typing import List
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time


class GeminiEmbeddingManager:
    """CPU-only embedding manager using Gemini's embedding API"""

    def __init__(self):
        self.device = "cpu"  # Always CPU for this implementation
        self.batch_size = 16  # Smaller batches for API calls
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini's embedding model
        self.model_name = "models/text-embedding-004"  # Latest Gemini embedding model
        
        print(f"Initialized Gemini embedding manager (CPU-only)")
        print(f"Using model: {self.model_name}")

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts using Gemini's embedding API with batching and rate limiting
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Process in smaller batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch_sync(batch)
            embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to respect rate limits
            if i + self.batch_size < len(texts):
                time.sleep(0.1)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if normalize:
            # L2 normalize the embeddings
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
        
        return embeddings_array

    def _encode_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous batch encoding using Gemini API
        """
        embeddings = []
        
        for text in texts:
            try:
                # Generate embedding using Gemini
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
            except Exception as e:
                print(f"Warning: Failed to generate embedding for text (length {len(text)}): {e}")
                # Fallback: create a zero embedding with the expected dimension
                # Gemini text-embedding-004 produces 768-dimensional embeddings
                embeddings.append([0.0] * 768)
                
        return embeddings

    async def encode_batch_async(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Async wrapper for batch encoding
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                self.encode_batch, 
                texts, 
                normalize
            )
        return result


class CPUOptimizedEmbeddingManager:
    """
    Fallback CPU-only embedding manager using SentenceTransformers
    Used as backup if Gemini API is unavailable
    """

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.batch_size = 16  # Smaller batches for CPU

        # Use a lightweight model optimized for CPU
        model_name = "all-MiniLM-L6-v2"  # Smaller and faster than L12-v2

        print(f"Loading CPU embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                model_name, device=self.device, trust_remote_code=True
            )
            print("✅ CPU embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CPU embedding model: {e}")
            raise

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """CPU-optimized batch encoding"""
        if not self.model:
            raise RuntimeError("Embedding model not available")
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device,
        )
        return embeddings.astype(np.float32)


def create_embedding_manager(prefer_gemini: bool = True):
    """
    Factory function to create the appropriate embedding manager
    """
    if prefer_gemini:
        try:
            return GeminiEmbeddingManager()
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini embedding manager: {e}")
            print("🔄 Falling back to CPU SentenceTransformer model...")
            return CPUOptimizedEmbeddingManager()
    else:
        return CPUOptimizedEmbeddingManager()