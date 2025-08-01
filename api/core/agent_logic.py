# api/core/agent_logic.py
from typing import List
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from .vector_store import RequestKnowledgeBase
from .structured_data_extractor import QueryClassifier

# Corrected: Use a valid, fast model ID
llm = Gemini(
    id="gemini-1.5-flash-latest",
    temperature=0.0,
    max_tokens=150,
    timeout=10.0
)

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Ultra-optimized agent for RTX 4060 - sub-3-second responses
    """
    
    def lightning_search(query_text: str) -> List[str]:
        """Lightning-fast search with query optimization"""
        query_type, concepts = QueryClassifier.classify_query(query_text)
        
        if query_type == "comparison" and len(concepts) >= 2:
            results = []
            for concept in concepts[:2]:
                concept_results = knowledge_base.search(f"{query_text} {concept}", k=3)
                results.extend(concept_results)
            return list(dict.fromkeys(results))[:5]
        
        elif query_type == "factual":
            return knowledge_base.search(query_text, k=4)
        
        else:
            expanded = expand_query_fast(query_text)
            all_results = []
            for exp_query in expanded[:2]:
                results = knowledge_base.search(exp_query, k=3)
                all_results.extend(results)
            return list(dict.fromkeys(all_results))[:5]

    # Corrected: More concise instructions to reduce token usage
    instructions = """You are an expert Q&A system. Use the lightning_search tool to find context from the document. Answer the user's question directly in 1-2 sentences based ONLY on the provided search results. If the answer is not in the results, state: 'Information not available in the document.'"""

    agent = Agent(
        tools=[lightning_search],
        instructions=instructions,
        model=llm,
        debug_mode=False,
        reasoning=False,
        max_loops=1,
        show_tool_calls=False
    )

    try:
        response: RunResponse = await agent.arun(question)
        return response.content
    except Exception as e:
        direct_results = knowledge_base.search(question, k=3)
        if direct_results:
            return f"Based on the document: {direct_results[0][:200]}..."
        return "Unable to find relevant information in the document."

def expand_query_fast(query: str) -> List[str]:
    """Ultra-fast query expansion with minimal overhead"""
    base_query = query.lower()
    expansions = {query}  # Use a set to handle duplicates
    
    quick_synonyms = {
        "rate": "fee", "limit": "maximum", "require": "need",
        "allow": "permit", "benefit": "advantage"
    }
    
    # Corrected: Allow multiple expansions instead of breaking after one
    for original, synonym in quick_synonyms.items():
        if original in base_query:
            expanded = base_query.replace(original, synonym)
            if expanded != base_query:
                expansions.add(expanded.title())
    
    return list(expansions)[:2] # Max 2 queries