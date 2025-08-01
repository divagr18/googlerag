# api/core/vector_store.py
import numpy as np
import faiss  # This will be faiss-cpu on your system
from typing import List

class RequestKnowledgeBase:
    """
    MODIFIED FOR CPU-ONLY: A streamlined FAISS vector store optimized for CPU performance.
    All GPU-specific code has been removed to ensure Windows compatibility.
    """
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.chunks: List[str] = []
        self.faiss_index = None
        self.cache = {}

    def build(self, chunks: List[str]):
        """Builds a CPU-based FAISS index."""
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        dimension = self.model.get_sentence_embedding_dimension()

        print(f"Building CPU-based FAISS index with {len(chunks)} chunks...")
        
        # Encoding will happen on the CPU as configured in main.py
        embeddings = self._batch_encode(chunks)
        
        # Create a standard CPU index. IndexFlatL2 is a good default for accuracy.
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        print(f"✅ FAISS index built successfully on CPU ({self.faiss_index.ntotal} vectors).")

    def _batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encodes texts in batches to manage memory, using the CPU."""
        print(f"Encoding {len(texts)} chunks on CPU...")
        # The model will use the 'cpu' device as set in main.py
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True, # Show progress as CPU encoding can take time
            convert_to_numpy=True
        )
        return embeddings

    def search(self, query: str, k: int = 5) -> List[str]:
        """Performs a search on the CPU index and uses an efficient FIFO cache."""
        cache_key = f"{query}_{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.faiss_index is None:
            raise ValueError("Knowledge base not built yet.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = [self.chunks[i] for i in indices[0] if i != -1]
        
        # Use FIFO cache eviction
        if len(self.cache) > 100:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = results
        
        return results