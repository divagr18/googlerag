# api/core/agent_logic.py
import os
import asyncio
from typing import List
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from .vector_store import RequestKnowledgeBase
from .structured_data_extractor import QueryClassifier
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM once
llm = Gemini(
    id="gemini-1.5-flash",
    temperature=0.0, # Set to 0.0 for more deterministic query generation
    api_key=GEMINI_API_KEY
)

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Uses an agent with three parallel search calls to maximize context coverage,
    while reusing the original optimized search logic.
    """
    
    async def lightning_search(query_text: str) -> List[str]:
        """
        Lightning-fast search with robust query optimization and symmetric prefixing.
        """
        query_type, concepts = QueryClassifier.classify_query(query_text)
        
        if query_type == "comparison" and len(concepts) >= 2:
            results = []
            prefixed_base_query = f"search_query: {query_text}"
            for concept in concepts[:2]:
                final_query = f"{prefixed_base_query} {concept}"
                concept_results = await knowledge_base.search(final_query, k=5) # Await async search
                results.extend(concept_results)
            return list(dict.fromkeys(results))[:3]
        
        elif query_type == "factual":
            prefixed_query = f"search_query: {query_text}"
            return await knowledge_base.search(prefixed_query, k=5) # Await async search
        
        else: # General query
            expanded_queries = expand_query_fast(query_text)
            all_results = []
            for exp_query in expanded_queries:
                prefixed_exp_query = f"search_query: {exp_query}"
                results = await knowledge_base.search(prefixed_exp_query, k=5) # Await async search
                all_results.extend(results)
            return list(dict.fromkeys(all_results))[:3]

    async def search_direct(query: str) -> List[str]:
        """Primary search. Use a direct, keyword-focused version of the user's question."""
        print(f"⚡️ [Search] Running Direct Search for: '{query}'")
        return await lightning_search(query)

    async def search_rephrased(hypothetical_question: str) -> List[str]:
        """Secondary search. Rephrase the user's question as a hypothetical 'what if' or 'how does' scenario to find related concepts."""
        print(f"⚡️ [Search] Running Rephrased Search for: '{hypothetical_question}'")
        return await lightning_search(hypothetical_question)

    async def search_detailed(specific_query: str) -> List[str]:
        """Tertiary search. Formulate a query to find specific details, numbers, definitions, or exceptions related to the main question."""
        print(f"⚡️ [Search] Running Detailed Search for: '{specific_query}'")
        return await lightning_search(specific_query)

    instructions = """You are an expert Q&A system. Your goal is to answer questions accurately and concisely based ONLY on the provided search results.

**CRITICAL RULES:**
1.  If a user asks for something that cannot be in a document (e.g., "write code", "generate a javascript function"), you MUST respond with: "I cannot answer this question as it is outside the scope of document analysis." Do not use the search tools or any other rule.
2.  Your final answer MUST be 2-3 sentences maximum. Be direct and to the point.

**Query Generation Rules:**
For all other questions, you MUST use all three search tools in parallel. Generate three distinct queries based on the user's question:
1.  `search_direct`: A short, keyword-focused query.
2.  `search_rephrased`: A full question that rephrases the user's intent.
3.  `search_detailed`: A specific question asking for details or lists.

**Answer Synthesis Rules:**
After receiving the search results, you MUST follow these rules to generate your final answer:
1.  **Synthesize, Don't Quote:** Combine information from the search results to form a coherent answer. Do not just copy-paste long passages.
2.  **Handle Missing Information:**
    - If search results partially answer the question, explain what is available in a brief, helpful way (e.g., "The document does not state X, but it does mention Y.").
    - If search results are completely irrelevant, respond ONLY with: "Information not available in the document."
3.  **Handle "Yes/No" Questions:** Start your answer with "Yes," or "No," followed by a brief, one-sentence explanation based on the context.
"""

    agent = Agent(
        tools=[search_direct, search_rephrased, search_detailed],
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
            return f"Based on a direct search of the document: {direct_results[0][:250]}..."
        return "Unable to process the request due to an internal error."

def expand_query_fast(query: str) -> List[str]:
    """Ultra-fast query expansion with minimal overhead"""
    base_query = query.lower()
    expansions = {query}
    quick_synonyms = {"rate": "fee", "limit": "maximum", "require": "need", "allow": "permit", "benefit": "advantage"}
    for original, synonym in quick_synonyms.items():
        if original in base_query:
            expanded = base_query.replace(original, synonym)
            if expanded != base_query: expansions.add(expanded.title())
    return list(expansions)[:2]