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
        return knowledge_base.search(query_text, k=8)

    # Define the instructions for the agent.
    agent_instructions = """
    You are an expert Q&A system. Your task is to answer the user's question based ONLY on the information you find using the `search_document_clauses` tool.
    1. First, use the `search_document_clauses` tool with a query that is relevant to the user's question.
    2. Then, carefully analyze the retrieved text from the tool's output.
    3. Formulate a direct and concise answer based *only* on the retrieved information. Give specific answers, prefer numbers and facts.
    4. If the retrieved text does not contain the answer, you MUST state that you could not find the information in the provided document. Do not use any external knowledge.
    5. Do not write too long answers. Be VERY concise and EXTREMELY to the point. Maximum 2 sentences. Do not add more info than needed. Final answer must be token usage efficient. Do not add any special formatting or markdown, only plain text.
    """

    # Create the Agno agent instance
    agent = Agent(
        tools=[search_document_clauses],
        instructions=agent_instructions,
        model=llm,
        debug_mode=False,
        reasoning=False  # Fast, capable, and cost-effective
        # Note: Agno automatically uses the OPENAI_API_KEY from the environment
    )

    # Run the agent asynchronously. `arun` is the async version of `run`.
    response_object: RunResponse = await agent.arun(question)
    
    return response_object.content