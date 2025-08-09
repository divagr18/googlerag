# LLM-Powered Query–Retrieval System

Natural language queries to extract answers from large unstructured documents (policies, contracts, emails).  
Uses hybrid retrieval (BM25 + FAISS) with Large Language Models to extract sourced answers from multi-format documents.

---

## Features
- Multi-format ingestion: PDF, DOCX, PPTX, XLSX, TXT, Images (OCR)
- Direct mode for small documents via Agno agent (low latency)
- Hybrid retrieval pipeline for large documents:
  - BM25 + FAISS search
  - Reciprocal Rank Fusion (RRF) and GPU-accelerated reranking
- Provenance in answers (page numbers and chunk text)
- Query decomposition and expansion for multi-aspect queries

---

## How It Works
1. Document ingestion: Streams and extracts text from uploaded documents (PDF, Word, PPT, Excel, images with OCR).
2. Chunking and embedding: Splits documents into semantic chunks and generates vector embeddings using SentenceTransformers (GPU-optimized).
3. Hybrid retrieval: Combines BM25 (keyword) and FAISS (semantic) search, fusing results with RRF.
4. Reranking: Uses a CrossEncoder model to rank top chunks for query relevance.
5. LLM synthesis: Sends top-ranked chunks to an LLM (Gemini/OpenAI) with an evidence-only prompt to generate an answer.
6. Provenance: Returns answers with source page numbers and original text snippets.

---

## Installation
```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

## Running the API
```bash
uvicorn api.main:app --reload
```

---

## API Usage
**Endpoint:**  
```
POST api/v1/hackrx/run
```

**Headers:**
```
Authorization: Bearer <your-token>
Content-Type: application/json
```

**Request JSON:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the policy for sick leave?",
    "Who are the signatories of the contract?"
  ]
}
```

**Response Example:**
```json
{
  "answers": [
    "Employees are entitled to 12 paid sick leave days per year."
  ]
}
```

---

## Environment Variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_gemini_key
OPENAI_API_KEY=your_openai_key
```

---

## Tech Stack
- Backend: FastAPI
- LLMs: Google Gemini, OpenAI GPT
- Search: FAISS (vector), BM25 (keyword)
- Embeddings and Reranking: SentenceTransformers (all-MiniLM-L6-v2), CrossEncoder (ms-marco)
- Document Processing: PyMuPDF, python-docx, python-pptx, pytesseract
- Other: asyncio, PyTorch (GPU), rank_bm25

---

## Folder Structure
```
api/
  core/
    agent_logic.py           # Orchestrates retrieval and synthesis
    agno_agent.py            # Direct small-doc mode
    document_processor.py    # File streaming, parsing, chunking
    embedding_manager.py     # GPU-optimized embeddings
    vector_store.py          # Hybrid BM25 + FAISS store
  routes/
    hackrx.py                # Main API route
requirements.txt
```

---

## Example Workflow
1. Start API:
   ```bash
   uvicorn api.main:app --reload
   ```
2. Send POST request to `api/v1/hackrx/run` with `documents` and `questions`.
3. Receive answers.

---

## Future Enhancements
- UI dashboard for document uploads and querying
- Persistent vector database (Milvus, Weaviate)
- Multi-tenant access controls
- Monitoring and analytics dashboard


