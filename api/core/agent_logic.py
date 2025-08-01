# api/core/agent_logic.py
from typing import List
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from .vector_store import RequestKnowledgeBase

# Use a fast and cost-effective model
llm = Gemini(id="gemini-1.5-flash-latest", temperature=0.0)

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    An agent that uses a search tool to find context, solving the
    "Inefficient LLM Token Usage" bottleneck.
    """
    
    def search_document(query_text: str) -> List[str]:
        """A tool for the agent to search the document knowledge base."""
        print(f"Agent searching for: '{query_text}'")
        return knowledge_base.search(query_text, k=3)

    # FIX: A concise prompt that tells the agent to use tools and be brief.
    # This dramatically reduces token cost.
    instructions = """You are a helpful Q&A assistant. Your goal is to answer questions based on a document.
    1. Use the `search_document` tool with a clear, concise query to find relevant information.
    2. Analyze the search results.
    3. Provide a direct, concise answer to the user's question based ONLY on the information from the search results.
    4. If the search results do not contain the answer, you MUST respond with 'Information not available in the document.'
    """

    agent = Agent(
        tools=[search_document],
        instructions=instructions,
        model=llm,
        max_loops=2, # Allow for a search and then a final answer
    )

    try:
        response: RunResponse = await agent.arun(question)
        return response.content
    except Exception as e:
        print(f"Agent execution failed: {e}")
        return "An error occurred while processing the question."