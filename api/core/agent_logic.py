# api/core/agent_logic.py  (New name for clarity)
from typing import List
from agno.agent import Agent
from agno.agent import RunResponse
from agno.models.google import Gemini
from .vector_store import RequestKnowledgeBase
from .structured_data_extractor import QueryClassifier

# Ultra-fast model configuration for RTX 4060
llm = Gemini(
    id="gemini-2.0-flash-exp",  # Latest fastest model
    temperature=0.0,
    max_tokens=150,  # Reduced for faster generation
    timeout=10.0     # Quick timeout
)

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Ultra-optimized agent for RTX 4060 - sub-3-second responses
    """
    
    def lightning_search(query_text: str) -> List[str]:
        """Lightning-fast search with query optimization"""
        # Pre-classify query for optimized search strategy
        query_type, concepts = QueryClassifier.classify_query(query_text)
        
        if query_type == "comparison" and len(concepts) >= 2:
            # Multi-concept search for comparisons
            results = []
            for concept in concepts[:2]:  # Limit to prevent slowdown
                concept_results = knowledge_base.search(f"{query_text} {concept}", k=3)
                results.extend(concept_results)
            return list(dict.fromkeys(results))[:5]  # Dedupe and limit
        
        elif query_type == "factual":
            # Precise search for factual queries
            return knowledge_base.search(query_text, k=4)
        
        else:
            # Standard search with query expansion
            expanded = expand_query_fast(query_text)
            all_results = []
            for exp_query in expanded[:2]:  # Limit expansions
                results = knowledge_base.search(exp_query, k=3)
                all_results.extend(results)
            return list(dict.fromkeys(all_results))[:5]

    # Minimal, high-performance instructions
    instructions = """
    You are a fast, precise document Q&A system.
    
    PROCESS:
    1. Search using lightning_search with the user's question
    2. Find the answer in the retrieved text
    3. Give a direct answer in 1-2 sentences
    
    If not found: "Information not available in the document."
    Be concise and accurate.
    """

    agent = Agent(
        tools=[lightning_search],
        instructions=instructions,
        model=llm,
        debug_mode=False,
        reasoning=False,  # Disable reasoning for speed
        max_loops=1,      # Single loop only
        show_tool_calls=False
    )

    try:
        response: RunResponse = await agent.arun(question)
        return response.content
    except Exception as e:
        # Fallback to direct search if agent fails
        direct_results = knowledge_base.search(question, k=3)
        if direct_results:
            return f"Based on the document: {direct_results[0][:200]}..."
        return "Unable to find relevant information in the document."

def expand_query_fast(query: str) -> List[str]:
    """Ultra-fast query expansion with minimal overhead"""
    base_query = query.lower()
    expansions = [query]  # Original first
    
    # Quick synonym replacement - only most common cases
    quick_synonyms = {
        "rate": "fee", "limit": "maximum", "require": "need",
        "allow": "permit", "benefit": "advantage"
    }
    
    for original, synonym in quick_synonyms.items():
        if original in base_query:
            expanded = base_query.replace(original, synonym)
            if expanded != base_query:
                expansions.append(expanded.title())  # Proper case
                break  # Only one expansion for speed
    
    return expansions[:2]  # Max 2 queries