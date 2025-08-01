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
    id="gemini-2.5-flash",
    temperature=0.0, # Set to 0.0 for more deterministic query generation
    api_key=GEMINI_API_KEY
)

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Uses an agent with three parallel search calls to maximize context coverage,
    while reusing the original optimized search logic.
    """
    
    # --- 1. RESTORED: Your original, optimized search logic is kept here ---
    # This function will be called by the new parallel tools.
    async def lightning_search(query_text: str) -> List[str]:
        """
        Lightning-fast search with robust query optimization and symmetric prefixing.
        """
        # Do NOT prefix the query_text variable here.
        # Instead, apply the prefix right before each call to the knowledge base.

        # 1. Classify the original, clean query text.
        query_type, concepts = QueryClassifier.classify_query(query_text)
        
        if query_type == "comparison" and len(concepts) >= 2:
            results = []
            # The base query is prefixed first.
            prefixed_base_query = f"search_query: {query_text}"
            for concept in concepts[:2]:
                # The concept is appended to the already prefixed query for a more specific search.
                final_query = f"{prefixed_base_query} {concept}"
                concept_results = knowledge_base.search(final_query, k=5)
                results.extend(concept_results)
            # Return a deduplicated list of the top 3 results.
            return list(dict.fromkeys(results))[:3]
        
        elif query_type == "factual":
            # Apply the prefix directly to the factual query.
            prefixed_query = f"search_query: {query_text}"
            # A slightly larger k is good for specific factual questions.
            return knowledge_base.search(prefixed_query, k=5)
        
        else: # General query
            # 2. Expand the original, clean query text.
            expanded_queries = expand_query_fast(query_text)
            all_results = []
            for exp_query in expanded_queries:
                # 3. CRITICAL FIX: Apply the prefix to EACH expanded query individually.
                prefixed_exp_query = f"search_query: {exp_query}"
                results = knowledge_base.search(prefixed_exp_query, k=5)
                all_results.extend(results)
            # Return a deduplicated list of the top 3 results from all expansions.
            return list(dict.fromkeys(all_results))[:3]

    # --- 2. NEW: Define three distinct wrapper tools for parallel execution ---
    # Each tool has a unique docstring to guide the LLM in creating varied queries.
    # They all call your original `lightning_search` function under the hood.
    
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

    # --- 3. NEW: Update instructions to use the three parallel tools ---
    instructions = """You are an expert Q&A system. Your primary goal is to answer the user's question accurately and relevantly based ONLY on the provided search results.

    If their query is something like code generation, skip the query generation and directly provide the code snippet.

**Query Generation Rules:**
You MUST use all three search tools in parallel. Generate three distinct queries based on the user's question:
1.  `search_direct`: A short, keyword-focused query (3-6 words). Example: "Article 12 State definition".
2.  `search_rephrased`: A full question that rephrases the user's intent. Example: "How is the term 'State' defined for fundamental rights?".
3.  `search_detailed`: A specific question asking for details, lists, or exceptions. Example: "What specific entities are included in the definition of 'State' under Article 12?".

**Answer Synthesis Rules:**
After receiving the search results, you MUST follow these rules to generate your final answer:
1.  **Answer the Core Question:** Directly address the user's question. Your main priority is to provide the information they asked for.
2.  **Be Relevant, Not Verbose:** Answer in 2-3 clear and focused sentences. Do not include information that is related but not directly asked for.
3.  **Synthesize, Don't Just State "Not Found":**
    - If the search results contain information that *partially* or *indirectly* answers the question, synthesize a helpful response based on what IS available.
    - **Example:** If asked "Is an arrest without a warrant legal?", and the text only describes the rights of an arrested person, a good answer is: "The document does not state whether an arrest without a warrant is legal, but it outlines the rights of an arrested person, such as the right to be informed of the grounds for arrest and to be produced before a magistrate within 24 hours."
    - Only if the search results are completely irrelevant to the question, respond with: "Information not available in the document."
4.  **Handle "Yes/No" Questions:** Begin your answer with "Yes," or "No," followed by a concise explanation based on the context.
"""

    # --- 4. Configure the agent with the new parallel tools ---
    agent = Agent(
        tools=[search_direct, search_rephrased, search_detailed],
        instructions=instructions,
        model=llm,
        debug_mode=False,
        reasoning=False,
    )

    try:
        # The agent will internally handle the parallel tool calls and synthesize the final answer.
        response: RunResponse = await agent.arun(question)
        
        # Directly return the final content generated by the agent.
        # The agent's instructions already tell it to say "Information not available..." if needed.
        return response.content

    except Exception as e:
        # This is a robust fallback for unexpected errors (e.g., API failure, agent crash).
        print(f"Agent failed with an unexpected error: {e}. Performing a direct, single search.")
        direct_results = await lightning_search(question)
        if direct_results:
            # Provide a snippet from the best result as a last resort.
            return f"Based on a direct search of the document: {direct_results[0][:250]}..."
        return "Unable to process the request due to an internal error."

def expand_query_fast(query: str) -> List[str]:
    """Ultra-fast query expansion with minimal overhead"""
    base_query = query.lower()
    expansions = {query}  # Use a set to handle duplicates
    
    quick_synonyms = {
        "rate": "fee", "limit": "maximum", "require": "need",
        "allow": "permit", "benefit": "advantage"
    }
    
    for original, synonym in quick_synonyms.items():
        if original in base_query:
            expanded = base_query.replace(original, synonym)
            if expanded != base_query:
                expansions.add(expanded.title())
    
    return list(expansions)[:2] # Max 2 queries