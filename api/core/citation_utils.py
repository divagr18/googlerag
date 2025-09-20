# api/core/citation_utils.py

from typing import List, Dict, Tuple, Set
from urllib.parse import urlparse
import re

class CitationManager:
    """
    Handles citation extraction and formatting for document chunks.
    Generates clickable hyperlinks to source documents with page references.
    """
    
    @staticmethod
    def format_pdf_page_link(url: str, page_number: int) -> str:
        """
        Format a PDF URL with page fragment for direct navigation.
        
        Args:
            url: Original PDF URL
            page_number: Page number to link to
            
        Returns:
            URL with page fragment (e.g., document.pdf#page=5)
        """
        if page_number and page_number > 0:
            return f"{url}#page={page_number}"
        return url
    
    @staticmethod
    def format_web_page_link(url: str, page_number: int = None) -> str:
        """
        Format a web page URL. For web pages, page numbers are less relevant.
        
        Args:
            url: Original web page URL
            page_number: Page number (usually not applicable for web pages)
            
        Returns:
            Original URL (web pages don't typically have page fragments)
        """
        return url
    
    @staticmethod
    def get_citation_link(url: str, page_number: int, file_type: str) -> str:
        """
        Get the appropriate citation link based on file type.
        
        Args:
            url: Document URL
            page_number: Page number
            file_type: File extension/type
            
        Returns:
            Formatted URL with appropriate page reference
        """
        file_type = file_type.lower()
        
        if file_type in ['pdf']:
            return CitationManager.format_pdf_page_link(url, page_number)
        elif file_type in ['docx', 'doc', 'pptx', 'ppt']:
            # For Office documents, we can't easily link to specific pages
            # but we include the page info in the citation text
            return url
        else:
            # For web pages and other formats
            return CitationManager.format_web_page_link(url, page_number)
    
    @staticmethod
    def extract_citations_from_chunks(chunks_with_metadata: List[Dict]) -> List[Dict]:
        """
        Extract citation information from chunks with metadata.
        
        Args:
            chunks_with_metadata: List of chunk dictionaries with metadata
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen_sources = set()  # To avoid duplicate citations
        
        for chunk in chunks_with_metadata:
            metadata = chunk.get('metadata', {})
            
            # Create a unique identifier for this source
            source_key = (
                metadata.get('document_url', ''),
                metadata.get('page_number', 1)
            )
            
            # Skip if we've already seen this exact source
            if source_key in seen_sources:
                continue
                
            seen_sources.add(source_key)
            
            url = metadata.get('document_url', '')
            page_number = metadata.get('page_number', 1)
            file_type = metadata.get('file_type', 'unknown')
            document_title = metadata.get('document_title', 'Unknown Document')
            
            if url:  # Only create citation if we have a URL
                citation_link = CitationManager.get_citation_link(url, page_number, file_type)
                
                citations.append({
                    'url': url,
                    'link': citation_link,
                    'page_number': page_number,
                    'file_type': file_type,
                    'document_title': document_title,
                    'source_key': source_key
                })
        
        return citations
    
    @staticmethod
    def format_citation_text(citation: Dict) -> str:
        """
        Format a single citation as readable text.
        
        Args:
            citation: Citation dictionary
            
        Returns:
            Formatted citation string
        """
        title = citation['document_title']
        page = citation['page_number']
        file_type = citation['file_type'].upper()
        
        if page and page > 1:
            return f"{title} ({file_type}, Page {page})"
        else:
            return f"{title} ({file_type})"
    
    @staticmethod
    def format_citation_markdown(citation: Dict) -> str:
        """
        Format a single citation as Markdown link.
        
        Args:
            citation: Citation dictionary
            
        Returns:
            Markdown formatted citation
        """
        text = CitationManager.format_citation_text(citation)
        link = citation['link']
        return f"[{text}]({link})"
    
    @staticmethod
    def format_citation_html(citation: Dict) -> str:
        """
        Format a single citation as HTML link.
        
        Args:
            citation: Citation dictionary
            
        Returns:
            HTML formatted citation
        """
        text = CitationManager.format_citation_text(citation)
        link = citation['link']
        return f'<a href="{link}" target="_blank">{text}</a>'
    
    @staticmethod
    def format_citations_section(
        citations: List[Dict], 
        format_type: str = "text",
        max_citations: int = 5
    ) -> str:
        """
        Format multiple citations into a sources section.
        
        Args:
            citations: List of citation dictionaries
            format_type: "text", "markdown", or "html"
            max_citations: Maximum number of citations to include
            
        Returns:
            Formatted citations section
        """
        if not citations:
            return ""
        
        # Limit citations and sort by page number
        limited_citations = sorted(
            citations[:max_citations], 
            key=lambda x: (x['document_title'], x['page_number'])
        )
        
        if format_type == "markdown":
            formatter = CitationManager.format_citation_markdown
            separator = "\n- "
            prefix = "\n\n**Sources:**\n- "
        elif format_type == "html":
            formatter = CitationManager.format_citation_html
            separator = "<br>• "
            prefix = "<br><br><strong>Sources:</strong><br>• "
        else:  # text format
            formatter = CitationManager.format_citation_text
            separator = "\n• "
            prefix = "\n\nSources:\n• "
        
        formatted_citations = [formatter(citation) for citation in limited_citations]
        
        # Add "and X more" if we truncated the list
        if len(citations) > max_citations:
            remaining = len(citations) - max_citations
            formatted_citations.append(f"and {remaining} more source{'s' if remaining > 1 else ''}")
        
        return prefix + separator.join(formatted_citations)
    
    @staticmethod
    def add_citations_to_answer(
        answer: str, 
        chunks_with_metadata: List[Dict],
        format_type: str = "text",
        max_citations: int = 3
    ) -> str:
        """
        Add numbered citations to an answer based on the chunks used.
        
        Args:
            answer: The generated answer text
            chunks_with_metadata: Chunks used to generate the answer
            format_type: Citation format ("text", "markdown", "html")
            max_citations: Maximum number of citations to include
            
        Returns:
            Answer with numbered citations [1] [2] and sources section appended
        """
        citations = CitationManager.extract_citations_from_chunks(chunks_with_metadata)
        
        if not citations:
            return answer
        
        # Limit citations
        limited_citations = citations[:max_citations]
        
        # Add numbered citations distributed throughout the answer text
        sentences = answer.split('. ')
        modified_sentences = []
        citation_index = 0
        
        # Distribute citations across sentences
        for i, sentence in enumerate(sentences):
            modified_sentence = sentence
            
            # Add citation every few sentences, ensuring all citations are used
            if citation_index < len(limited_citations):
                # Strategy: place citations in first sentence, middle sentences, and last sentence
                should_add_citation = (
                    i == 0 or  # First sentence
                    i == len(sentences) - 1 or  # Last sentence
                    (len(sentences) > 2 and i == len(sentences) // 2)  # Middle sentence
                )
                
                # If we still have citations left and we're running out of sentences
                if citation_index < len(limited_citations) - 1 and i >= len(sentences) - (len(limited_citations) - citation_index):
                    should_add_citation = True
                
                if should_add_citation and citation_index < len(limited_citations):
                    # Add citation number to the sentence
                    if not sentence.endswith('.'):
                        modified_sentence += f" [{citation_index + 1}]."
                    else:
                        modified_sentence = sentence[:-1] + f" [{citation_index + 1}]."
                    citation_index += 1
                elif not sentence.endswith('.') and i < len(sentences) - 1:
                    modified_sentence += "."
            elif not sentence.endswith('.') and i < len(sentences) - 1:
                modified_sentence += "."
            
            modified_sentences.append(modified_sentence)
        
        # Join sentences back
        answer_with_numbers = ' '.join(modified_sentences)
        
        # Create numbered sources section
        sources_lines = []
        for i, citation in enumerate(limited_citations, 1):
            title = citation['document_title']
            page = citation['page_number']
            file_type = citation['file_type'].upper()
            
            if page and page > 1:
                source_line = f"[{i}] {title} ({file_type}, Page {page})"
            else:
                source_line = f"[{i}] {title} ({file_type})"
            sources_lines.append(source_line)
        
        # Format the sources section
        if format_type == "markdown":
            sources_section = "\n\n**Sources:**\n• " + "\n• ".join(sources_lines)
        elif format_type == "html":
            sources_section = "<br><br><strong>Sources:</strong><br>• " + "<br>• ".join(sources_lines)
        else:  # text format
            sources_section = "\n\nSources:\n• " + "\n• ".join(sources_lines)
        
        return answer_with_numbers + sources_section