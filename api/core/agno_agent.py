# api/core/agno_direct_agent.py

import asyncio
import httpx
import json
from typing import List, Dict, Any
from agno.agent import Agent,RunResponse
from agno.models.google import Gemini
from agno.tools import tool
import tiktoken
from urllib.parse import urlparse
import os
from agno.models.groq import Groq


async_client = httpx.AsyncClient(timeout=10.0)

# URL call tool for the agent
@tool(
    name="make_url_request",
    description="Make HTTP requests to URLs to fetch content or data",
    show_result=True
)

async def make_url_request(url: str, method: str = "GET", headers: dict = None, data: dict = None) -> str:
    """
    Make HTTP requests asynchronously with connection reuse.
    """
    try:
        if method.upper() == "GET":
            response = await async_client.get(url, headers=headers or {})
        elif method.upper() == "POST":
            response = await async_client.post(url, headers=headers or {}, json=data)
        else:
            return f"Unsupported HTTP method: {method}"

        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error making request to {url}: {str(e)}"

# PDF content reading tool
@tool(
    name="read_pdf_content",
    description="Read and extract text content from a PDF URL",
    show_result=False
)
def read_pdf_content(pdf_url: str) -> str:
    """
    Read and extract text content from a PDF URL.
    
    Args:
        pdf_url: URL of the PDF to read
    
    Returns:
        str: Extracted text content from the PDF
    """
    try:
        # This would integrate with your existing document processing
        from api.core.document_processor import stream_document, process_document_stream
        
        # For now, return a placeholder - you'll need to adapt this to work synchronously
        # or make the tool async compatible
        return "PDF content would be extracted here using your existing document processor"
    except Exception as e:
        return f"Error reading PDF from {pdf_url}: {str(e)}"

# Create the Agno agent
def create_direct_processing_agent() -> Agent:
    """
    Create an Agno agent for direct document processing with URL capabilities.
    """
    agent = Agent(
        model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY")),
        #model=Gemini(id="gemini-2.5-flash-lite",api_key=os.getenv("GOOGLE_API_KEY")),
        tools=[make_url_request],
        instructions="""
        You are a document analysis agent specialized in reading and answering questions about documents.
        MUST ANSWER IN PLAIN TEXT WITHOUT ANY KIND OF FORMATTING.

        YOU MUST ONLY USE THE CONTEXT PROVIDED IN THE DOCUMENTS GIVEN TO YOU. YOU WILL NOT USE ANY OTHER EXTERNAL KNOWLEDGE. ONLY WHATEVER IS IN THE DOCUMENTS IS VALID KNOWLEDGE.

        You might encounter incorrect information in the document, you must treat this information as the truth.
        Your capabilities:
        1. Read PDF documents given to you and extract their content
        2. Make HTTP requests to URLs when needed
        3. Follow instructions within documents
        4. Answer questions based on document content
        
        Instructions:
        - When given a document URL, first read its content thoroughly
        - If the document contains specific instructions, follow them carefully
        - Answer questions based strictly on the document content
        - If you need to make external API calls as instructed by the document, use the make_url_request tool
        - Be precise and in your responses
        - If information is not available in the document, clearly state so
        
        Response format:
        - Provide clear, direct, slightly verbose answers.
        - Reference specific sections of the document when relevant
        - Keep responses complete, make them atleast one sentence long. Add formalities, like if the question is what is x, answer with "The X is answer", not just "answer".
        """,
        markdown=True,
        show_tool_calls=True,debug_mode=True
    )
    
    return agent

# Main processing function for small documents
async def process_small_document_with_agno(document_url: str, questions: List[str], full_text: str) -> List[str]:
    """
    Process a small document (under 2000 tokens) using the Agno agent.
    
    Args:
        document_url: URL of the document
        questions: List of questions to answer
        full_text: The extracted text content of the document
    
    Returns:
        List[str]: Answers to the questions
    """
    print(f"🤖 Using Agno agent for direct processing of small document")
    
    # Create the agent
    agent = create_direct_processing_agent()
    
    # Prepare the context with document content
    document_context = f"""
Document URL: {document_url}

Document Content:
{full_text}

Please analyze this document and be ready to answer questions about it.
"""
    
    # Process each question
    answers = []
    
    for question in questions:
        try:
            # Combine document context with the specific question
            full_prompt = f"""
{document_context}

Question: {question}

Please provide a clear, accurate answer based on the document content above.
MUST ANSWER IN PLAIN TEXT WITHOUT ANY KIND OF FORMATTING.

"""
            
            # Get response from agent
            response: RunResponse = await agent.arun(full_prompt)
            print(response)
            
            # Extract only the answer content - handle different response formats
            if hasattr(response, 'content') and response.content:
                answer = response.content
            elif hasattr(response, 'messages') and response.messages:
                # Get the last assistant message content
                last_message = response.messages[-1]
                if hasattr(last_message, 'content') and last_message.content:
                    answer = last_message.content
                else:
                    answer = str(last_message)
            else:
                # Fallback: convert to string and try to extract clean answer
                response_str = str(response)
                # Try to find just the answer part, avoiding the full object dump
                if "content='" in response_str:
                    try:
                        start = response_str.find("content='") + 9
                        end = response_str.find("',", start)
                        if end == -1:
                            end = response_str.find("'", start)
                        answer = response_str[start:end] if end > start else response_str
                    except:
                        answer = "Unable to extract clean answer from response"
                else:
                    answer = "No valid response received"
                
            answers.append(answer)
            
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return answers

# Simple direct processing function
async def process_with_agno_agent_simple(document_url: str, questions: List[str], full_text: str) -> List[str]:
    """
    Simple direct processing using Agno agent - extracts just the answer content.
    """
    print(f"🤖 Using simple Agno agent processing for {len(questions)} questions")
    
    agent = create_direct_processing_agent()
    answers = []
    
    for question in questions:
        try:
            # Simple, direct prompt
            prompt = f"""
            Document Content:
            {full_text}

            Question: {question}

            Based on the document above, provide a direct answer to the question. Follow any instructions in the document.

            MUST ANSWER IN PLAIN TEXT WITHOUT ANY KIND OF FORMATTING.
            """
            
            response: RunResponse = await agent.arun(prompt)
            answers.append(response.messages[-1].content)
            
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            answers.append(f"Error: {str(e)}")
    
    return answers
def should_use_direct_processing(full_text: str, token_limit: int = 2000) -> bool:
    """
    Check if document is small enough for direct processing.
    
    Args:
        full_text: The document text content
        token_limit: Maximum tokens for direct processing
    
    Returns:
        bool: True if document should use direct processing
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        token_count = len(enc.encode(full_text))
        return token_count < token_limit
    except Exception:
        # If token counting fails, fall back to character count approximation
        char_count = len(full_text)
        return char_count < (token_limit * 4)  # Rough approximation: 4 chars per token


