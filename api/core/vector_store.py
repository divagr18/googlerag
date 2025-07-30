import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi # Import the BM25 library

class RequestKnowledgeBase:
    """
    An in-memory, per-request knowledge base supporting hybrid search
    with FAISS (semantic) and BM25 (keyword).
    """
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.chunks: List[str] = []
        
        # Vector search component
        self.faiss_index = None
        
        # Keyword search component
        self.bm25_index = None

    def build(self, chunks: List[str]):
        """Builds both the FAISS and BM25 indexes from a list of text chunks."""
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        print("Building knowledge base...")

        # --- 1. Build Keyword (BM25) Index ---
        tokenized_corpus = [doc.split(" ") for doc in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("✅ BM25 keyword index built.")

        # --- 2. Build Vector (FAISS) Index ---
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_np)
        
        dimension = embeddings_np.shape[1]
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
        self.faiss_index.add(embeddings_np)
        print(f"✅ FAISS vector index built with {self.faiss_index.ntotal} vectors.")
        print("✅ Knowledge base ready for hybrid search.")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Performs hybrid search using Reciprocal Rank Fusion (RRF).
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("Knowledge base has not been built yet.")
        query = "search_query: " + query
        print(f"Performing hybrid search for query: '{query}'")

        # --- 1. Keyword Search (BM25) ---
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        # Get top k*2 results to ensure good overlap
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_results = {idx: score for idx, score in zip(bm25_top_indices, bm25_scores[bm25_top_indices])}

        # --- 2. Vector Search (FAISS) ---
        query_embedding = self.model.encode([query])
        query_embedding_np = np.array(query_embedding, dtype='float32')
        faiss.normalize_L2(query_embedding_np)
        
        distances, faiss_top_indices = self.faiss_index.search(query_embedding_np, k*2)
        faiss_results = {idx: dist for idx, dist in zip(faiss_top_indices[0], distances[0])}

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        # `k` is a constant used in the RRF formula, 60 is a common value.
        rrf_k = 60
        fused_scores: Dict[int, float] = {}

        # Process BM25 results
        for rank, idx in enumerate(bm25_top_indices):
            if idx not in fused_scores:
                fused_scores[idx] = 0.0
            fused_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # Process FAISS results
        for rank, idx in enumerate(faiss_top_indices[0]):
            if idx == -1: continue # Skip invalid indices
            if idx not in fused_scores:
                fused_scores[idx] = 0.0
            fused_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # --- 4. Re-rank based on fused scores ---
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Return the top k unique chunks
        top_k_indices = [idx for idx, score in reranked_results[:k]]
        
        return [self.chunks[i] for i in top_k_indices]