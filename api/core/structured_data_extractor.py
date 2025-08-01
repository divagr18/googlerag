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
        
        # Extract key concepts (nouns and important terms)
        concepts = cls._extract_concepts(query_lower)
        
        # Classify query type
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
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        # Extract words that are likely to be important concepts
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        concepts = []
        
        for word in words:
            word_lower = word.lower()
            if (word_lower not in stop_words and 
                len(word_lower) > 2 and 
                not word_lower in ['what', 'how', 'when', 'where', 'why', 'who']):
                concepts.append(word_lower)
        
        return concepts[:5]  # Return top 5 concepts

# Additional utility functions for structured data extraction
class DocumentAnalyzer:
    """Utility class for document structure analysis"""
    
    @staticmethod
    def extract_structured_elements(text: str) -> dict:
        """Extract structured elements like tables, lists, etc."""
        elements = {
            'tables': [],
            'lists': [],
            'sections': [],
            'key_terms': []
        }
        
        # Extract numbered lists
        list_pattern = r'^\s*(?:\d+\.|\*|-)\s+(.+)$'
        elements['lists'] = re.findall(list_pattern, text, re.MULTILINE)
        
        # Extract section headers
        header_pattern = r'^[A-Z][A-Z\s]{10,}$'
        elements['sections'] = re.findall(header_pattern, text, re.MULTILINE)
        
        # Extract percentage/rate patterns
        rate_pattern = r'\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:percent|rate|fee)\b'
        elements['key_terms'] = re.findall(rate_pattern, text, re.IGNORECASE)
        
        return elements
