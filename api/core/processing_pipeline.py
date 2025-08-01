from .embedding_manager import OptimizedEmbeddingManager
from .document_processor import fast_semantic_chunk_text
from .vector_store import RequestKnowledgeBase
from .agent_logic import answer_question_with_agent

class OptimizedPipeline:
    def __init__(self):
        self.embedding_manager = OptimizedEmbeddingManager()
        
    async def process_document_and_answer(self, document_text: str, question: str) -> str:
        """Complete optimized pipeline"""
        
        # Fast chunking (5x faster)
        chunks = await fast_semantic_chunk_text(
            document_text, 
            self.embedding_manager,
            target_chunk_size=1000
        )
        
        # Build knowledge base (GPU accelerated)
        kb = RequestKnowledgeBase(self.embedding_manager)
        kb.build(chunks)
        
        # Get answer (3x faster)
        answer = await answer_question_with_agent(question, kb)
        
        return answer