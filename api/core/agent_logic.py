# api/core/agent_logic.py
import os
import asyncio
import time
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
query_enhancer_llm = Gemini(
    id="gemini-2.5-flash-lite",
    temperature=0.3,
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
- Keep each query concise
- Focus on terms likely to appear in documents
- For comparison queries, ensure both items are covered
- For numerical queries, include terms that might appear near numbers
- Avoid question words (what, how, when) - focus on content terms

Format your response as:
DIRECT: [query]
EXPANDED: [query] 
CONTEXTUAL: [query]"""

    try:
        # --- TIMER START ---
        t0 = time.perf_counter()
        
        query_agent = Agent(
            model=query_enhancer_llm,
            instructions="You are a search query expert. Follow the user's instructions exactly.",
            debug_mode=False,
            reasoning=False
        )
        
        response: RunResponse = await query_agent.arun(enhancement_prompt)
        
        # --- TIMER END ---
        t1 = time.perf_counter()
        print(f"    ⏱️ LLM call for '{original_query[:30]}...' took {t1 - t0:.2f}s")

        enhanced_queries = parse_enhanced_queries(response.content)
        
        if not enhanced_queries or len(enhanced_queries) != 3:
            print(f"    ⚠️ LLM query enhancement failed for '{original_query[:30]}...', falling back to rules.")
            return generate_fallback_queries(original_query, query_type)
            
        return enhanced_queries
        
    except Exception as e:
        print(f"    ⚠️ Query enhancement error for '{original_query[:30]}...': {e}, using fallback")
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
    # ... (no changes to this function)
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

async def prepare_enhanced_queries_for_all_questions(questions: List[str]) -> List[Dict[str, str]]:
    """
    Pre-process all questions to generate enhanced queries in parallel.
    Returns a list of enhanced query dictionaries for each question.
    """
    print(f"🚀 Pre-processing {len(questions)} questions for enhanced queries...")
    
    # --- TIMER START ---
    t_start = time.perf_counter()

    async def process_single_question(question: str) -> Dict[str, str]:
        query_type, concepts = QueryClassifier.classify_query(question)
        enhanced_queries = await enhance_query_with_llm(question, query_type)
        print(f"  ✅ Enhanced queries for '{question[:50]}...': {query_type}")
        return {
            'question': question,
            'query_type': query_type,
            'concepts': concepts,
            'enhanced_queries': enhanced_queries
        }
    
    tasks = [process_single_question(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    # --- TIMER END ---
    t_end = time.perf_counter()
    
    print(f"🎯 All query enhancements completed in {t_end - t_start:.2f}s!")
    return results
async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase, 
                                   precomputed_queries: Dict[str, str] = None) -> str:
    """
    Uses an agent with pre-computed enhanced queries for maximum performance.
    """
    
    # Use precomputed queries if available, otherwise compute on-the-fly
    if precomputed_queries:
        query_type = precomputed_queries['query_type']
        concepts = precomputed_queries['concepts']
        enhanced_queries = precomputed_queries['enhanced_queries']
        print(f"🔄 Using precomputed queries for: {question[:50]}...")
    else:
        # Fallback to original behavior
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
                concept_results = await knowledge_base.search(final_query, k=5, fusion_weights=fusion_weights)
                results.extend(concept_results)
            return list(dict.fromkeys(results))[:3]
        
        elif query_type == "factual":
            prefixed_query = f"search_query: {query_text}"
            return await knowledge_base.search(prefixed_query, k=5, fusion_weights=fusion_weights)
        
        else: # General query
            prefixed_query = f"search_query: {query_text}"
            return await knowledge_base.search(prefixed_query, k=5, fusion_weights=fusion_weights)

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
    instructions = f""""You are a specialized AI system that functions as a factual database for a given document. Your primary role is to answer a user's question by first using a set of search tools to find relevant evidence, and then synthesizing that evidence into a concise, factual answer.

**The user's question is:** "{question}"

**--- YOUR TWO-STEP PROCESS ---**

**STEP 1: SEARCH FOR EVIDENCE**

To gather evidence, you MUST execute all three of the following search tools in parallel. These tools have been pre-configured with optimized queries based on the user's question.

1. `search_direct`: Retrieves results using a keyword-focused query.
2. `search_expanded`: Retrieves results using a query with related concepts and synonyms.
3. `search_contextual`: Retrieves results using a query that considers the broader context.

*Exception Rule:* If the user's question is clearly not something that can be answered from a document (e.g., "write code", "give me a password"), you MUST skip the search step and immediately respond with the exact phrase: "I cannot answer this question as it is outside the scope of document analysis."

**STEP 2: SYNTHESIZE THE FINAL ANSWER**

After you have received the results from all three searches, you MUST synthesize a final answer based on the following strict rules.

**--- ANSWER SYNTHESIS RULES ---**

1.  **BE A DATABASE, NOT A CHATBOT:** Your tone must be impersonal and factual.
    - DO NOT use conversational phrases like "The policy states..." or "As you can see...".
    - Get straight to the point.

2.  **SUMMARIZE, DO NOT QUOTE:** Never quote the evidence directly. Read the evidence, understand the rule, and state the rule in your own words as a concise summary.

3.  **QUOTE RELEVANT CLAUSES/ARTICLES**: Must quote specific articles, clauses, or sections from the document that directly support your answer. Use the exact text from the document, but do not quote it as a direct citation. Instead, summarize the rule in your own words.

4.  **HANDLE MISSING INFORMATION:** If the search results do not contain the information to answer the question, You may use your knowledge to fill in some gaps, such as if it is asked "If the police torture someone in custody, what right is being violated?", you may use your knowledge to identify the right being violated, but only if you cannot find relevant information in the document, and only if you are confident in your answer.
    You must be detailed enough if doing so, 2-3 sentences max. Quote articles, clauses if applicable. If you still cannot answer that at all, respond with the single, exact phrase: "I could not find relevant information in the document."

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