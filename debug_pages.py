#!/usr/bin/env python3

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.core.chroma_manager import ChromaDocumentManager

def debug_page_numbers():
    """Debug function to check page numbers in ChromaDB"""
    try:
        manager = ChromaDocumentManager()
        
        # Get all documents
        docs = manager.list_documents()
        print(f"Found {len(docs)} documents")
        
        for doc in docs:
            if 'quant' in doc['document_title'].lower():
                print(f"\n=== Document: {doc['document_title']} ===")
                print(f"Document ID: {doc['document_id']}")
                
                # Get some chunks for this document
                try:
                    results = manager.collection.get(
                        where={"document_id": doc['document_id']},
                        limit=10
                    )
                    
                    print(f"Found {len(results['ids'])} chunks")
                    
                    # Check page numbers in metadata
                    for i, (chunk_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
                        page_num = metadata.get('page_number', 'Unknown')
                        print(f"Chunk {i+1}: Page {page_num}")
                        
                        if i >= 5:  # Limit output
                            break
                            
                except Exception as e:
                    print(f"Error getting chunks: {e}")
                    
                break
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_page_numbers()