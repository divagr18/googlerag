import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class RequestKnowledgeBase:
    """
    CPU-optimized knowledge base with intelligent caching and memory management
    """
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.chunks: List[str] = []
        self.faiss_index: faiss.IndexFlatIP = None
        self.bm25_index: BM25Okapi = None
        self.cache: Dict[str, List[str]] = {}

    def build(self, chunks: List[str]):
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        print(f"Building CPU-optimized KB with {len(chunks)} chunks...")

        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [self._fast_tokenize(doc) for doc in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Build CPU FAISS index
        print("Building CPU FAISS index...")
        embeddings = self._batch_encode(chunks)
        
        # --- MODIFICATION: Ensure embeddings are float32 for CPU FAISS ---
        if embeddings.dtype == np.float16:
            print("ℹ️  [KB Build] Converting float16 embeddings to float32 for FAISS CPU index.")
            embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        del embeddings
        print(f"✅ CPU-optimized KB ready ({self.faiss_index.ntotal} vectors)")

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        # --- MODIFIED: Conditional batch size ---
        # Use a larger batch size for GPU, smaller for CPU
        batch_size = 64 if self.model.device.type == 'cuda' else 16
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=len(batch), # Use the actual size of the current batch
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.model.device
            )
            all_embeddings.append(embeddings)
            
        final_embeddings = np.vstack(all_embeddings)
        
        # --- ADDED: Print dimensions, data type, and batch size ---
        print(
            f"ℹ️  [KB Build] Generated embeddings: "
            f"Shape={final_embeddings.shape}, "
            f"DType={final_embeddings.dtype}, "
            f"Batch Size={batch_size}"
        )
        
        return final_embeddings

    def search(self, query: str, k: int = 5) -> List[str]:
        cache_key = f"{query}_{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("Knowledge base not built yet.")

        search_k = min(k * 2, len(self.chunks), 20)
        bm25_results = self._bm25_search(query, search_k)
        faiss_results = self._faiss_search(query, search_k)
        
        fused = self._fuse_results(bm25_results, faiss_results, query)
        results = fused[:k]

        # FIFO cache eviction
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = results
        return results

    def _fast_tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        tokens = self._fast_tokenize(query)
        if not tokens:
            return {}
        scores = self.bm25_index.get_scores(tokens)
        if len(scores) <= k:
            return {i: s for i, s in enumerate(scores) if s > 0}
        top = np.argpartition(scores, -k)[-k:]
        return {i: scores[i] for i in top if scores[i] > 0}

    def _faiss_search(self, query: str, k: int) -> Dict[int, float]:
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.model.device
        )
        # --- MODIFICATION: Ensure query embedding is float32 for search ---
        if emb.dtype == np.float16:
            emb = emb.astype(np.float32)
            
        distances, indices = self.faiss_index.search(emb, k)
        return {idx: float(dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1 and dist > 0}

    def _fuse_results(self, bm25: Dict[int, float], faiss: Dict[int, float], query: str) -> List[str]:
        # Normalize
        if bm25:
            max_b = max(bm25.values())
            bm25 = {i: v / max_b for i, v in bm25.items()}
        if faiss:
            max_f = max(faiss.values())
            faiss = {i: v / max_f for i, v in faiss.items()}
        
        # Weighting
        weights = (0.4, 0.6)
        indices = set(bm25) | set(faiss)
        scored = []
        for i in indices:
            score = weights[0] * bm25.get(i, 0) + weights[1] * faiss.get(i, 0)
            if score > 0:
                scored.append((i, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, _ in scored]

    def __del__(self):
        pass