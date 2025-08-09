# **LLM-Powered Intelligent Query–Retrieval System**

Natural language → Accurate answers from large unstructured documents (policies, contracts, emails).  
Uses hybrid retrieval (BM25 + FAISS) with Large Language Models to extract precise, sourced answers from multi-format documents.

---

## **Features**
- Multi-format ingestion: **PDF, DOCX, PPTX, XLSX, TXT, Images (with OCR)**
- Small-doc direct mode via **Agno agent** (low latency)
- Large-doc hybrid RAG pipeline:
  - **BM25 + FAISS** hybrid search
  - Reciprocal Rank Fusion (RRF) + **GPU-accelerated reranking**
- Provenance in every answer (page numbers + chunk text)
- Query decomposition & expansion for multi-aspect queries

---

## **How It Works**
1. **Document ingestion** – The system streams and extracts text from uploaded documents (PDF, Word, PPT, Excel, images with OCR).  
2. **Smart chunking & embedding** – Breaks large documents into semantic chunks and generates vector embeddings using SentenceTransformers (GPU-optimized).  
3. **Hybrid retrieval** – Combines BM25 (keyword) and FAISS (semantic) search, fusing results with RRF for better recall.  
4. **Reranking** – Uses a CrossEncoder model to rank top chunks for query relevance.  
5. **LLM synthesis** – Sends top-ranked chunks to an LLM (Gemini/OpenAI) with an evidence-only prompt to generate an accurate answer.  
6. **Provenance** – Returns answers with source page numbers and original text snippets.  

---

## **Installation**
```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

## **Running the API**
```bash
uvicorn api.main:app --reload
```
By default runs at: **http://127.0.0.1:8000**

---

## **API Usage**
**Endpoint:**  
```
POST /hackrx/run
```

**Headers:**
```
Authorization: Bearer <your-token>
Content-Type: application/json
```

**Request JSON format:**
```json
{
  "doc": "https://example.com/document.pdf",
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
    {
      "question": "What is the policy for sick leave?",
      "answer": "Employees are entitled to 12 paid sick leave days per year.",
      "sources": [
        {
          "page": 3,
          "text": "Employees are entitled to 12 paid sick leave days..."
        }
      ]
    }
  ]
}
```

---

## **Environment Variables**
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_gemini_key
OPENAI_API_KEY=your_openai_key
BEARER_TOKEN=your_api_token
```

---

## **Tech Stack**
- **Backend:** FastAPI
- **LLMs:** Google Gemini, OpenAI GPT
- **Search:** FAISS (vector) + BM25 (keyword)
- **Embeddings & Reranking:** SentenceTransformers (`all-MiniLM-L6-v2`), CrossEncoder (`ms-marco`)
- **Document Processing:** PyMuPDF, python-docx, python-pptx, pytesseract
- **Other:** asyncio, PyTorch (GPU acceleration), rank_bm25

---

## **Folder Structure**
```
api/
  core/
    agent_logic.py           # Orchestrates retrieval & synthesis
    agno_direct_agent.py     # Direct small-doc mode
    document_processor.py    # File streaming, parsing, chunking
    embedding_manager.py     # GPU-optimized embeddings
    vector_store.py          # Hybrid BM25 + FAISS store
  routes/
    hackrx.py                 # Main API route
requirements.txt
```

---

## **Example Workflow**
1. **Start API:**
   ```bash
   uvicorn api.main:app --reload
   ```
2. **Send POST request** to `/hackrx/run` with `doc` and `questions`.
3. **Receive answers** with evidence and page numbers.

---

## **Future Enhancements**
- UI dashboard for drag-and-drop document uploads & querying
- Persistent vector DB (e.g., Milvus, Weaviate)
- Multi-tenant access controls
- Live monitoring and analytics dashboard
