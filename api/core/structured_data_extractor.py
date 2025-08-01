import re
from typing import List, Tuple

class QueryClassifier:
    """
    Fast query classification for optimized retrieval strategies
    """
    
    COMPARISON_PATTERNS = [
        r'\b(difference|differ|compare|comparison|versus|vs\.?|between)\b',
        r'\b(similar|same|identical|equivalent)\b',
        r'\b(better|worse|higher|lower|more|less)\b than',
        r'\b(both|either|neither)\b'
    ]
    
    FACTUAL_PATTERNS = [
        r'\b(what|how much|how many|when|where|who)\b',
        r'\b(rate|percentage|amount|fee|cost|price)\b',
        r'\b(requirement|condition|eligibility|criteria)\b'
    ]
    
    CONDITIONAL_PATTERNS = [
        r'\b(if|when|unless|provided|assuming|given)\b',
        r'\b(can I|am I|do I|should I|must I)\b',
        r'\b(allowed|permitted|eligible|qualified)\b'
    ]
    
    @classmethod
    def classify_query(cls, query: str) -> Tuple[str, List[str]]:
        """
        Fast query classification with concept extraction
        Returns: (query_type, key_concepts)
        """
        query_lower = query.lower()
        concepts = cls._extract_concepts(query_lower)
        
        if any(re.search(pattern, query_lower) for pattern in cls.COMPARISON_PATTERNS):
            return "comparison", concepts
        elif any(re.search(pattern, query_lower) for pattern in cls.CONDITIONAL_PATTERNS):
            return "conditional", concepts
        elif any(re.search(pattern, query_lower) for pattern in cls.FACTUAL_PATTERNS):
            return "factual", concepts
        else:
            return "general", concepts
    
    @classmethod
    def _extract_concepts(cls, query: str) -> List[str]:
        """Extract key concepts from query"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        concepts = []
        for word in words:
            word_lower = word.lower()
            if (word_lower not in stop_words and 
                len(word_lower) > 2 and 
                not word_lower in ['what', 'how', 'when', 'where', 'why', 'who']):
                concepts.append(word_lower)
        
        return concepts[:5]

# Removed the unused DocumentAnalyzer class