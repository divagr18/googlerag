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

# URL call tool for the agent
@tool(
    name="make_url_request",
    description="Make HTTP requests to URLs to fetch content or data",
    show_result=True
)
def make_url_request(url: str, method: str = "GET", headers: Dict[str, str] = None, data: Dict[str, Any] = None) -> str:
    """
    Make HTTP requests to URLs and return the response content.
    
    Args:
        url: The URL to make the request to
        method: HTTP method (GET, POST, etc.)
        headers: Optional headers to include
        data: Optional data to send with request
    
    Returns:
        str: The response content or error message
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=headers or {})
            elif method.upper() == "POST":
                response = client.post(url, headers=headers or {}, json=data)
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
        model=Gemini(id="gemini-2.5-flash-lite",api_key=os.getenv("GOOGLE_API_KEY")),
        tools=[make_url_request, read_pdf_content],
        instructions="""
        You are a document analysis agent specialized in reading and answering questions about documents.
        MUST ANSWER IN PLAIN TEXT WITHOUT ANY KIND OF FORMATTING.

        
        Your capabilities:
        1. Read PDF documents and extract their content
        2. Make HTTP requests to URLs when needed
        3. Follow instructions within documents
        4. Answer questions based on document content
        
        Instructions:
        - When given a document URL, first read its content thoroughly
        - If the document contains specific instructions, follow them carefully
        - Answer questions based strictly on the document content
        - If you need to make external API calls as instructed by the document, use the make_url_request tool
        - Be precise and factual in your responses
        - If information is not available in the document, clearly state so
        
        Response format:
        - Provide clear, direct answers
        - Reference specific sections of the document when relevant
        - Keep responses concise but complete
        """,
        markdown=True,
        show_tool_calls=True
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
            response: RunResponse = agent.run(full_prompt)
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
            
            response: RunResponse = agent.run(prompt)
            print(response.messages[-1].content)
            
            # Extract just the content we need
            if hasattr(response, 'content'):
                clean_answer = str(response.content).strip()
            else:
                # Try to get the actual answer from the response object
                response_str = str(response)
                
                # Look for common answer patterns
                if "flight number is" in response_str.lower():
                    import re
                    match = re.search(r'flight number is \*\*([^*]+)\*\*', response_str, re.IGNORECASE)
                    if match:
                        clean_answer = match.group(1)
                    else:
                        match = re.search(r'flight number is ([a-zA-Z0-9]+)', response_str, re.IGNORECASE)
                        clean_answer = match.group(1) if match else "Could not extract flight number"
                else:
                    # General content extraction
                    clean_answer = "Unable to extract answer"
            
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

# Async wrapper for the agent (since Agno agents might be sync)
async def run_agno_agent_async(agent: Agent, prompt: str) -> str:
    """
    Run Agno agent asynchronously.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: agent.run(prompt))

# Enhanced processing function with better error handling
async def process_with_agno_agent_enhanced(document_url: str, questions: List[str], full_text: str) -> List[str]:
    """
    Enhanced processing using Agno agent with better error handling and parallel processing.
    """
    print(f"🤖 Using enhanced Agno agent processing for {len(questions)} questions")
    
    agent = create_direct_processing_agent()
    
    # Prepare base context
    base_context = f"""
You have access to the following document:

URL: {document_url}
Content:
{full_text}

Follow any instructions in the document and answer questions based on its content.
"""
    
    # Process questions in parallel for better performance
    async def process_single_question(question: str) -> str:
        try:
            full_prompt = f"{base_context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Use the async wrapper
            response = await run_agno_agent_async(agent, full_prompt)
            
            # Clean response extraction
            if hasattr(response, 'content') and response.content:
                return response.content
            elif hasattr(response, 'messages') and response.messages:
                # Get the last assistant message content
                last_message = response.messages[-1]
                if hasattr(last_message, 'content') and last_message.content:
                    return last_message.content
                else:
                    return str(last_message)
            else:
                # Fallback: try to extract clean answer from string representation
                response_str = str(response)
                # Look for the actual answer content, avoiding metadata
                if "The flight number is" in response_str:
                    # Extract just the relevant answer part
                    lines = response_str.split('\n')
                    for line in lines:
                        if "flight number is" in line.lower():
                            return line.strip()
                
                # General fallback for content extraction
                if "content='" in response_str:
                    try:
                        start = response_str.find("content='") + 9
                        end = response_str.find("',", start)
                        if end == -1:
                            end = response_str.find("'", start)
                        if end > start:
                            return response_str[start:end]
                    except:
                        pass
                
                return "Unable to extract clean answer from response"
            
        except Exception as e:
            print(f"Error processing question '{question[:50]}...': {e}")
            return f"I encountered an error while processing this question: {str(e)}"
    
    # Process all questions concurrently
    answers = await asyncio.gather(*[
        process_single_question(q) for q in questions
    ])
    
    return answers