# api/core/agent_logic.py  (New name for clarity)
from typing import List
from agno.agent import Agent
from functools import partial
from agno.agent import RunResponse
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = OpenAIChat(id="gpt-4.1-nano", temperature=0.1)  # Fast, capable, and cost-effective
llm = Gemini(id="gemini-2.5-flash", temperature=0.1, api_key=GEMINI_API_KEY)  # Fast, capable, and cost-effective
# Import the Knowledge Base class
from .vector_store import RequestKnowledgeBase

async def answer_question_with_agent(question: str, knowledge_base: RequestKnowledgeBase) -> str:
    """
    Creates and runs a temporary Agno agent to answer a single question
    using the pre-built knowledge base.
    """
    
    # Define the tool for the agent. This tool searches our specific,
    # per-request knowledge base.
    def search_document_clauses(query_text: str) -> List[str]:
        """
        Use this tool to find relevant clauses or text from the document
        to answer the user's question.
        """
        print(f"Agent tool 'search_document_clauses' called with query: '{query_text}'")
        return knowledge_base.search(query_text, k=18)

    # Define the instructions for the agent.
    agent_instructions = """
    You are an expert Q&A system. Your task is to answer the user's question based ONLY on the information retrieved using the search_document_clauses tool.
    Use the search_document_clauses tool with a query that is precise, complete, and directly aligned with the user's question.
    Carefully analyze all retrieved clauses for factual content.
    Your answer must be strictly based on the retrieved content — no assumptions, no external knowledge. If the content does not provide a direct answer, state that the answer is not available in the document. If the answer is a yes/no question, begin with "Yes" or "No" based on the retrieved content.
    Include all relevant facts: waiting periods, exclusions, exceptions, special conditions (e.g., accident-related waivers), clause-specific rules, and all numerical details (percentages, durations, limits, counts, thresholds).
    If the document mentions alternate scenarios or triggers that override the general rule (e.g., accident, emergency), you MUST include them.
    If the information is not found in the retrieved content, explicitly state that the answer is not available in the document.
    Your final answer must be short, dense, and precise — maximum 2 sentences. Do not restate the question. Avoid explanations, qualifiers, or formatting. No markdown formatting, escaping quotes etc.
    """

    # Create the Agno agent instance
    agent = Agent(
        tools=[search_document_clauses],
        instructions=agent_instructions,
        model=llm,
        debug_mode=True,
        reasoning=False  # Fast, capable, and cost-effective
        # Note: Agno automatically uses the OPENAI_API_KEY from the environment
    )

    # Run the agent asynchronously. `arun` is the async version of `run`.
    response_object: RunResponse = await agent.arun(question)
    
    return response_object.content