# api/core/query_expander.py

import re
import asyncio
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

class DomainQueryExpander:
    def __init__(self, chunk_texts: List[str], min_term_freq: int = 3, max_expansion_terms: int = 5):
        self.min_term_freq = min_term_freq
        self.max_expansion_terms = max_expansion_terms
        self.stop_words = set(stopwords.words('english'))
        self.domain_terms: Set[str] = set()
        self.acronym_expansions: Dict[str, str] = {}
        self.term_cooccurrence: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.tfidf_vectorizer = None
        self.term_importance_scores: Dict[str, float] = {}
        
        print("🔍 Building domain-specific vocabulary...")
        self._build_vocabulary(chunk_texts)
        print(f"✅ Domain vocabulary built: {len(self.domain_terms)} terms, {len(self.acronym_expansions)} acronyms")

    def _build_vocabulary(self, chunk_texts: List[str]):
        all_text = " ".join(chunk_texts)
        self._extract_technical_terms(chunk_texts)
        self._build_tfidf_vocabulary(chunk_texts)
        self._extract_acronyms(all_text)
        self._build_cooccurrence_matrix(chunk_texts)

    def _extract_technical_terms(self, chunk_texts: List[str]):
        technical_term_patterns = [
            r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', r'\b[a-z]+(?:[A-Z][a-z]*)+\b',
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b', r'\b\w+[-_]\w+\b',
            r'\b\d+\.\d+(?:\.\d+)*\b', r'\b[A-Z]+\d+\b',
        ]
        all_terms = Counter()
        for text in chunk_texts:
            for pattern in technical_term_patterns:
                all_terms.update(re.findall(pattern, text))
            try:
                for chunk in ne_chunk(pos_tag(word_tokenize(text))):
                    if hasattr(chunk, 'label'):
                        entity = ' '.join(c[0] for c in chunk)
                        if len(entity) > 2: all_terms[entity] += 1
            except Exception:
                continue
        self.domain_terms = {term for term, freq in all_terms.items() if freq >= self.min_term_freq and term.lower() not in self.stop_words}

    def _build_tfidf_vocabulary(self, chunk_texts: List[str]):
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.8, stop_words='english')
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            for term, score in zip(feature_names, mean_scores):
                if score > 0.01:
                    self.term_importance_scores[term] = score
                    self.domain_terms.add(term)
        except Exception as e:
            print(f"⚠️ TF-IDF vocabulary building failed: {e}")

    def _extract_acronyms(self, text: str):
        patterns = [r'([A-Z][a-z\s]+)\s*\(([A-Z]{2,})\)', r'([A-Z]{2,})\s*\(([A-Z][a-z\s]+)\)']
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if len(match[0].split()) <= 5:
                    self.acronym_expansions[match[1].strip()] = match[0].strip()
                    self.acronym_expansions[match[0].strip()] = match[1].strip()

    def _build_cooccurrence_matrix(self, chunk_texts: List[str]):
        for text in chunk_texts:
            tokens = [t for t in re.findall(r'\b\w+\b', text.lower()) if t not in self.stop_words and len(t) > 2]
            for i, token in enumerate(tokens):
                if token in self.domain_terms or token in self.term_importance_scores:
                    window = tokens[max(0, i - 5):i] + tokens[i + 1:i + 6]
                    for cotoken in window:
                        if cotoken in self.domain_terms:
                            self.term_cooccurrence[token][cotoken] += 1
        for term, cooccur_counts in self.term_cooccurrence.items():
            total = sum(cooccur_counts.values())
            if total > 0:
                for coterm in cooccur_counts:
                    self.term_cooccurrence[term][coterm] /= total

    def expand_query(self, original_query: str, expansion_strategy: str = "hybrid") -> List[str]:
        query_variants = {original_query}
        query_terms = re.findall(r'\b\w+\b', original_query.lower())
        
        if expansion_strategy in ["acronym", "hybrid"]:
            query_variants.add(self._expand_with_acronyms(original_query))
        if expansion_strategy in ["cooccurrence", "hybrid"]:
            cooccur_expanded = self._expand_with_cooccurrence(query_terms)
            if cooccur_expanded: query_variants.add(cooccur_expanded)
        if expansion_strategy in ["tfidf", "hybrid"]:
            query_variants.add(self._expand_with_tfidf(original_query))
        
        return list(query_variants)[:4]

    def _expand_with_acronyms(self, query: str) -> str:
        for term, expansion in self.acronym_expansions.items():
            if term.lower() in query.lower():
                query = re.sub(r'\b' + re.escape(term) + r'\b', f"{term} {expansion}", query, flags=re.IGNORECASE)
        return query

    def _expand_with_cooccurrence(self, query_terms: List[str]) -> str:
        expansion_terms = set()
        for term in query_terms:
            if term in self.term_cooccurrence:
                top_cooccur = sorted(self.term_cooccurrence[term].items(), key=lambda x: x[1], reverse=True)[:2]
                for coterm, score in top_cooccur:
                    if score > 0.1 and coterm not in query_terms:
                        expansion_terms.add(coterm)
        if expansion_terms:
            return f"{' '.join(query_terms)} {' '.join(list(expansion_terms)[:self.max_expansion_terms])}"
        return ""

    def _expand_with_tfidf(self, query: str) -> str:
        if not self.tfidf_vectorizer: return query
        try:
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            expansion_candidates = []
            for term, importance in self.term_importance_scores.items():
                if term not in query.lower() and importance > 0.05:
                    term_words = set(term.lower().split())
                    if query_terms & term_words or any(qt in term for qt in query_terms):
                        expansion_candidates.append((term, importance))
            top_expansions = [term for term, _ in sorted(expansion_candidates, key=lambda x: x[1], reverse=True)[:3]]
            if top_expansions:
                return f"{query} {' '.join(top_expansions)}"
        except Exception as e:
            print(f"⚠️ TF-IDF expansion failed: {e}")
        return query