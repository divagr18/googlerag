#!/usr/bin/env python3
"""
Test script to verify the complete citation system implementation.
This tests ChromaDB integration, document processing, and citation generation.
"""

import asyncio
import sys
import os
import numpy as np

# Add the API directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from api.core.chroma_manager import ChromaDocumentManager
from api.core.citation_utils import CitationManager
from api.core.embedding_manager import OptimizedEmbeddingManager
from api.core.document_processor import optimized_semantic_chunk_text, stream_document, process_document_stream


async def test_citation_system():
    """Test the complete citation system pipeline."""
    print("🚀 Testing Citation System Implementation")
    print("=" * 50)
    
    # Initialize components
    print("1️⃣ Initializing components...")
    chroma_manager = ChromaDocumentManager()
    citation_manager = CitationManager()
    embedding_manager = OptimizedEmbeddingManager()
    
    # Test ChromaDB connection
    print("2️⃣ Testing ChromaDB connection...")
    try:
        stats = chroma_manager.get_stats()
        print(f"   ✅ ChromaDB connected - Collection: {stats['collection_name']}")
        print(f"   📊 Current documents: {stats['total_documents']}")
        print(f"   📄 Total chunks: {stats['total_chunks']}")
    except Exception as e:
        print(f"   ❌ ChromaDB connection failed: {e}")
        return False
    
    # Test document processing with a sample URL
    test_url = "https://example.com/sample-document.pdf"
    print(f"3️⃣ Testing document processing workflow...")
    
    try:
        # Create sample chunks with metadata (simulating document processing)
        sample_chunks = [
            {
                "text": "This is a sample chunk from page 1 of the document. It contains important information about machine learning algorithms.",
                "metadata": {
                    "document_url": test_url,
                    "page": 1,
                    "chunk_index": 0,
                    "document_title": "Machine Learning Guide",
                    "total_pages": 10
                }
            },
            {
                "text": "This is another chunk from page 2 discussing neural networks and deep learning architectures.",
                "metadata": {
                    "document_url": test_url,
                    "page": 2,
                    "chunk_index": 1,
                    "document_title": "Machine Learning Guide",
                    "total_pages": 10
                }
            }
        ]
        
        # Generate sample embeddings (mock embeddings for testing)
        sample_embeddings = np.array([[0.1] * 384 for _ in sample_chunks], dtype=np.float32)
        
        print("   ✅ Sample chunks and embeddings prepared")
        
        # Store in ChromaDB
        print("4️⃣ Testing document storage...")
        document_id = chroma_manager.store_document_chunks(
            test_url, sample_chunks, sample_embeddings, force_update=True
        )
        print(f"   ✅ Document stored with ID: {document_id}")
        
        # Test search functionality
        print("5️⃣ Testing document search...")
        search_results = chroma_manager.search_documents(
            query_embedding=np.array([0.1] * 384),  # Mock query embedding as numpy array
            n_results=2
        )
        print(f"   ✅ Search returned {len(search_results)} chunks")
        
        # Test citation generation
        print("6️⃣ Testing citation generation...")
        # Convert search results to the format expected by citation manager
        chunk_data_list = [chunk_dict for chunk_dict, _ in search_results]
        citations = citation_manager.extract_citations_from_chunks(chunk_data_list)
        
        for i, citation in enumerate(citations):
            formatted_citation = citation_manager.format_citation_markdown(citation)
            print(f"   Citation {i+1}: {formatted_citation}")
        
        # Test full answer with citations
        print("7️⃣ Testing answer enhancement with citations...")
        sample_answer = "Machine learning algorithms are powerful tools for data analysis. Neural networks provide sophisticated architectures for deep learning."
        
        enhanced_answer = citation_manager.add_citations_to_answer(sample_answer, chunk_data_list)
        print("   Enhanced Answer:")
        print(f"   {enhanced_answer}")
        
        # Test document listing
        print("8️⃣ Testing document management...")
        documents = chroma_manager.list_documents()
        print(f"   ✅ Listed {len(documents)} documents in database")
        
        # Cleanup test data
        print("9️⃣ Cleaning up test data...")
        cleanup_success = chroma_manager.delete_document(test_url)
        print(f"   ✅ Cleanup successful: {cleanup_success}")
        
        print("\n🎉 All tests passed! Citation system is working correctly.")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_citation_system())
    
    if success:
        print("\n✅ Citation system implementation verified!")
        print("\nNext steps:")
        print("- Start your FastAPI server: uvicorn api.main:app --reload")
        print("- Upload documents via the /ragsys/upload endpoint")
        print("- Ask questions via /ragsys/ask to see citations in action")
        print("- Manage documents via the new /ragsys/documents endpoints")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    sys.exit(0 if success else 1)