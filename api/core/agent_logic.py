from typing import List
from agno.agent import Agent
from agno.agent import RunResponse
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Choose your LLM (the last assignment wins)
llm = OpenAIChat(id="gpt-4.1-nano", temperature=0.1)
llm = Gemini(id="gemini-2.5-flash", temperature=0.1, api_key=GEMINI_API_KEY)

# Import the Knowledge Base class
from .vector_store import RequestKnowledgeBase

async def answer_question_with_agent(
    question: str,
    knowledge_base: RequestKnowledgeBase
) -> str:
    """
    Creates and runs a temporary Agno agent to answer a single question
    using the pre-built knowledge base.
    """

    def create_sync_search_tool(kb: RequestKnowledgeBase):
        """Synchronous wrapper for the knowledge-base search."""
        def sync_search(query_text: str) -> List[str]:
            print(f"Agent tool 'search_document_clauses' called with query: '{query_text}'")
            # kb.search is already synchronous—just call it directly
            return kb.search(query_text, k=30)
        return sync_search

    # Instantiate the synchronous tool
    search_document_clauses = create_sync_search_tool(knowledge_base)

    agent_instructions = """
    You are an expert Q&A system. Your task is to answer the user's question based ONLY on the
    information retrieved using the search_document_clauses tool.

    CRITICAL: Use the search_document_clauses tool with a query that is precise, complete, and
    directly aligned with the user's question.

    Steps:
    1. First, call search_document_clauses with keywords from the user's question
    2. Analyze the retrieved content carefully
    3. If no relevant information is found, try different search terms
    4. Base your answer ONLY on the retrieved content

    Answer Guidelines:
    - Your answer must be strictly based on the retrieved content — no assumptions, no external knowledge
    - If the content does not provide a direct answer, state that the answer is not available in the document
    - If it's a yes/no question, begin with "Yes" or "No" based on the retrieved content
    - Include all relevant facts: waiting periods, exclusions, exceptions, special conditions
    - If alternate scenarios or triggers exist (e.g., accident, emergency), include them
    - Be short, dense, and precise — maximum 2 sentences
    - No markdown formatting, no escaping quotes
    - Do not restate the question
    """

    agent = Agent(
        tools=[search_document_clauses],
        instructions=agent_instructions,
        model=llm,
        debug_mode=False,
        reasoning=False
    )

    try:
        response_object: RunResponse = await agent.arun(question)
        return response_object.content
    except Exception as e:
        print(f"Error running agent: {e}")
        return f"Error processing question: {str(e)}"
