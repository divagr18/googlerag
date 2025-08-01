import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

class OptimizedEmbeddingManager:
    """GPU-optimized embedding manager for RTX 4060"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.batch_size = 64 if self.device == "cuda" else 16
        
        # Use faster, smaller model optimized for your GPU
        model_name = "nomic-ai/nomic-embed-text-v1.5"  # 22MB, very fast on RTX 4060
        # Alternative: "paraphrase-MiniLM-L3-v2" for even faster processing
        
        print(f"Loading embedding model on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device,trust_remote_code=True)
        
        # Optimize for inference
        if self.device == "cuda":
            self.model.half()  # Use FP16 for 2x speed improvement on RTX 4060
            torch.backends.cudnn.benchmark = True
    
    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Optimized batch encoding with GPU acceleration"""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=self.device
            )
        return embeddings.astype(np.float32)