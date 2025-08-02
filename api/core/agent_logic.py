# api/core/agent_logic.py
import os
import asyncio
from typing import List, Dict, Tuple
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from .vector_store import RequestKnowledgeBase
from .structured_data_extractor import QueryClassifier
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM once
llm = Gemini(
    id="gemini-2.5-flash",
    temperature=0.1,
    api_key=GEMINI_API_KEY
)

# Separate LLM instance for query enhancement (lower temperature for consistency)
query_enhancer_llm = OpenAIChat(
    id="gpt-4.1-mini",
    temperature=0.2,  # More deterministic for query generation
    api_key=GEMINI_API_KEY
)

async def enhance_query_with_llm(original_query: str, query_type: str) -> Dict[str, str]:
    """
    Uses LLM to generate enhanced search queries based on query type.
    Returns exactly 3 enhanced queries to maintain compatibility.
    """
    
    enhancement_prompt = f"""You are a search query expert. Given a user's question, generate 3 different search queries that would help find the most relevant information in a document.

Original question: "{original_query}"
Query type: {query_type}

Generate exactly 3 enhanced queries:
1. DIRECT: A keyword-focused version that extracts the core terms
2. EXPANDED: An expanded version that includes related concepts and synonyms  
3. CONTEXTUAL: A version that considers broader context and implications

Guidelines:
- Keep each query concise (max 10 words)
- Focus on terms likely to appear in documents
- For comparison queries, ensure both items are covered
- For numerical queries, include terms that might appear near numbers
- Avoid question words (what, how, when) - focus on content terms

Format your response as:
DIRECT: [query]
EXPANDED: [query] 
CONTEXTUAL: [query]"""

    try:
        # Create a simple agent for query enhancement
        query_agent = Agent(
            model=query_enhancer_llm,
            instructions="You are a search query expert. Follow the user's instructions exactly.",
            tools=[],  # No tools needed for query enhancement
            debug_mode=False,
            reasoning=False
        )
        
        response: RunResponse = await query_agent.arun(enhancement_prompt)
        enhanced_queries = parse_enhanced_queries(response.content)
        
        # Fallback to original logic if parsing fails
        if not enhanced_queries or len(enhanced_queries) != 3:
            print("⚠️ LLM query enhancement failed, falling back to original logic")
            return generate_fallback_queries(original_query, query_type)
            
        return enhanced_queries
        
    except Exception as e:
        print(f"⚠️ Query enhancement error: {e}, using fallback")
        return generate_fallback_queries(original_query, query_type)

def parse_enhanced_queries(llm_response: str) -> Dict[str, str]:
    """Parse the LLM response into structured queries"""
    queries = {}
    lines = llm_response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('DIRECT:'):
            queries['direct'] = line.replace('DIRECT:', '').strip()
        elif line.startswith('EXPANDED:'):
            queries['expanded'] = line.replace('EXPANDED:', '').strip()
        elif line.startswith('CONTEXTUAL:'):
            queries['contextual'] = line.replace('CONTEXTUAL:', '').strip()
    
    return queries if len(queries) == 3 else {}

def generate_fallback_queries(original_query: str, query_type: str) -> Dict[str, str]:
    """Fallback query generation using original logic"""
    if query_type == "comparison":
        return {
            'direct': original_query,
            'expanded': f"{original_query} differences similarities",
            'contextual': f"{original_query} comparison analysis"
        }
    elif query_type == "factual":
        return {
            'direct': original_query,
            'expanded': f"{original_query} details information",
            'contextual': f"{original_query} requirements specifications"
        }
    else:
        expanded_queries = expand_query_fast(original_query)
        return {
            'direct': original_query,
            'expanded': expanded_queries[0] if len(expanded_queries) > 0 else original_query,
            'contextual': expanded_queries[1] if len(expanded_queries) > 1 else f"{original_query} context"
        }

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Uses an agent with LLM-enhanced queries for maximum accuracy.
    """
    
    # Step 1: Classify query and enhance with LLM
    query_type, concepts = QueryClassifier.classify_query(question)
    enhanced_queries = await enhance_query_with_llm(question, query_type)
    
    print(f"🔍 Query type: {query_type}")
    print(f"🚀 Enhanced queries: {enhanced_queries}")
    
    async def lightning_search(query_text: str, search_type: str = "general") -> List[str]:
        """
        Lightning-fast search with robust query optimization and symmetric prefixing.
        Now with enhanced fusion weights based on search type.
        """
        # Dynamic fusion weights based on search type and query type
        fusion_weights = get_dynamic_fusion_weights(query_type, search_type)
        
        if query_type == "comparison" and len(concepts) >= 2:
            results = []
            for concept in concepts[:2]:
                final_query = f"{query_text} {concept}"
                concept_results = await knowledge_base.search(final_query, k=3, fusion_weights=fusion_weights)
                results.extend(concept_results)
            return list(dict.fromkeys(results))[:3]
        
        elif query_type == "factual":
            prefixed_query = f"search_query: {query_text}"
            return await knowledge_base.search(prefixed_query, k=3, fusion_weights=fusion_weights)
        
        else: # General query
            prefixed_query = f"search_query: {query_text}"
            return await knowledge_base.search(prefixed_query, k=3, fusion_weights=fusion_weights)

    async def search_direct(query: str) -> List[str]:
        """Primary search using LLM-enhanced direct query."""
        enhanced_query = enhanced_queries.get('direct', query)
        print(f"⚡️ [Search] Running Direct Search for: '{enhanced_query}'")
        return await lightning_search(enhanced_query, "direct")

    async def search_expanded(query: str) -> List[str]:
        """Secondary search using LLM-enhanced expanded query."""
        enhanced_query = enhanced_queries.get('expanded', query)
        print(f"⚡️ [Search] Running Expanded Search for: '{enhanced_query}'")
        return await lightning_search(enhanced_query, "expanded")

    async def search_contextual(query: str) -> List[str]:
        """Tertiary search using LLM-enhanced contextual query."""
        enhanced_query = enhanced_queries.get('contextual', query)
        print(f"⚡️ [Search] Running Contextual Search for: '{enhanced_query}'")
        return await lightning_search(enhanced_query, "contextual")

    # Enhanced instructions with focus on accuracy over brevity
    instructions = f"""You are an expert Q&A system with access to document search tools. Your goal is to provide accurate, comprehensive answers based ONLY on the provided search results.

**CRITICAL RULES:**
1. If a user asks for something that cannot be in a document (e.g., "write code", "generate a javascript function"), you MUST respond with: "I cannot answer this question as it is outside the scope of document analysis." Do not use the search tools.

2. **Answer Quality Priority**: Provide complete, accurate answers. You must however limit yourself to 2-3 sentences per answer, but ensure you cover all relevant aspects of the question.

**Query Generation Rules:**
For document-related questions, you MUST use all three search tools in parallel, concurrently:
1. `search_direct`: Uses optimized keyword-focused search
2. `search_expanded`: Uses concept-expanded search with related terms  
3. `search_contextual`: Uses broader contextual search

Query Type Detected: {query_type}
Key Concepts: {concepts}

**Answer Synthesis Rules:**
1. **Comprehensive Coverage**: Address all aspects of the question based on available evidence. However, do not answer outside the scope of the question. Be concise but thorough.
2. **Evidence Integration**: Synthesize information from multiple search results into a coherent answer
3. **Handle Missing Information**: If search results partially answer the question, clearly explain what information is available and what might be missing
4. **Handle Irrelevant Results**: If search results are completely irrelevant, respond with "I could not find relevant information in the document."
5. **Yes/No Questions**: Start with "Yes," or "No," followed by a complete explanation with supporting evidence
6. **Confidence and Uncertainty**: Acknowledge when information is unclear or when you're making inferences
7. **Source Integration**: Explain how different pieces of evidence support your answer

"""

    agent = Agent(
        tools=[search_direct, search_expanded, search_contextual],
        instructions=instructions,
        model=llm,
        debug_mode=False,
        reasoning=False,
    )

    try:
        response: RunResponse = await agent.arun(question)
        return response.content
    except Exception as e:
        print(f"Agent failed with an unexpected error: {e}. Performing a direct, single search.")
        direct_results = await lightning_search(question)
        if direct_results:
            return f"Based on a direct search of the document: {direct_results[0][:500]}..."
        return "Unable to process the request due to an internal error."

def get_dynamic_fusion_weights(query_type: str, search_type: str) -> Tuple[float, float]:
    """
    Returns (bm25_weight, faiss_weight) based on query and search type.
    Higher BM25 weight for exact matching, higher FAISS for semantic similarity.
    """
    base_weights = {
        "factual": (0.6, 0.4),      # Favor exact term matching for facts
        "comparison": (0.4, 0.6),   # Favor semantic similarity for comparisons  
        "conditional": (0.5, 0.5),  # Balanced for conditional queries
        "general": (0.4, 0.6)       # Favor semantic for general queries
    }
    
    # Adjust based on search type
    bm25_weight, faiss_weight = base_weights.get(query_type, (0.4, 0.6))
    
    if search_type == "direct":
        # Direct searches should favor exact matching
        bm25_weight += 0.1
        faiss_weight -= 0.1
    elif search_type == "contextual":
        # Contextual searches should favor semantic similarity
        bm25_weight -= 0.1
        faiss_weight += 0.1
    
    # Ensure weights sum to 1.0
    total = bm25_weight + faiss_weight
    return (bm25_weight / total, faiss_weight / total)

def expand_query_fast(query: str) -> List[str]:
    """Ultra-fast query expansion with minimal overhead (kept for fallback)"""
    base_query = query.lower()
    expansions = {query}
    quick_synonyms = {"rate": "fee", "limit": "maximum", "require": "need", "allow": "permit", "benefit": "advantage"}
    for original, synonym in quick_synonyms.items():
        if original in base_query:
            expanded = base_query.replace(original, synonym)
            if expanded != base_query: expansions.add(expanded.title())
    return list(expansions)[:2]