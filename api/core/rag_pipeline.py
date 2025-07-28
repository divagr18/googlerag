# api/core/agent_logic.py  (New name for clarity)
from typing import List
from agno.agent import Agent
from functools import partial
from agno.models.openai import OpenAIChat
llm = OpenAIChat(model="gpt-4.1-mini", temperature=0.1)  # Fast, capable, and cost-effective

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
        return knowledge_base.search(query_text, k=5)

    # Define the instructions for the agent.
    agent_instructions = """
    You are an expert Q&A system. Your task is to answer the user's question based ONLY on the information you find using the `search_document_clauses` tool.
    1. First, use the `search_document_clauses` tool with a query that is relevant to the user's question.
    2. Then, carefully analyze the retrieved text from the tool's output.
    3. Formulate a direct and concise answer based *only* on the retrieved information.
    4. If the retrieved text does not contain the answer, you MUST state that you could not find the information in the provided document. Do not use any external knowledge.
    """

    # Create the Agno agent instance
    agent = Agent(
        tools=[search_document_clauses],
        instructions=agent_instructions,
        model=llm,
        debug_mode=True  # Fast, capable, and cost-effective
        # Note: Agno automatically uses the OPENAI_API_KEY from the environment
    )

    # Run the agent asynchronously. `arun` is the async version of `run`.
    response = await agent.arun(question)
    
    return response