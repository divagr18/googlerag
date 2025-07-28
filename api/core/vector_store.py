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
        
        # --- CHANGE 1: Encode directly to a NumPy array ---
        # No need for convert_to_tensor=True here. The default is NumPy.
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Ensure it's a float32 array, which FAISS prefers
        embeddings_np = np.array(embeddings, dtype='float32')
        
        # Normalize embeddings for cosine similarity (using Inner Product)
        faiss.normalize_L2(embeddings_np)
        
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
        
        # --- CHANGE 2: Encode the query directly to a NumPy array ---
        query_embedding = self.model.encode([query]) # The [query] ensures it's a 2D array
        
        # Ensure it's a float32 array
        query_embedding_np = np.array(query_embedding, dtype='float32')
        
        # Normalize the query vector
        faiss.normalize_L2(query_embedding_np)
        
        # Now we are guaranteed to be passing a correctly shaped NumPy array
        distances, indices = self.index.search(query_embedding_np, k)
        
        valid_indices = [i for i in indices[0] if i != -1]
        return [self.chunks[i] for i in valid_indices]