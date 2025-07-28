# api/core/vector_store.py
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer

class RequestKnowledgeBase:
    """
    An in-memory, per-request knowledge base using a FAISS-GPU index.
    """
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.index = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]):
        """Builds the FAISS index from a list of text chunks."""
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        print(f"Embedding {len(chunks)} chunks for the knowledge base...")
        embeddings = self.model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        
        # Normalize embeddings for cosine similarity (using Inner Product)
        faiss.normalize_L2(embeddings)
        
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            self.index = faiss.IndexFlatIP(dimension)
            
        self.index.add(embeddings_np)
        print(f"✅ Knowledge base ready with {self.index.ntotal} vectors.")

    def search(self, query: str, k: int = 5) -> List[str]:
        """Searches the knowledge base for the k most relevant chunks."""
        if self.index is None:
            raise ValueError("Knowledge base has not been built yet.")
        
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        faiss.normalize_L2(query_embedding)
        query_embedding_np = query_embedding.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Filter out invalid indices (-1) which can occur if k > number of vectors
        valid_indices = [i for i in indices[0] if i != -1]
        return [self.chunks[i] for i in valid_indices]