# Document Intelligence & Query System

An advanced document processing and question-answering system that combines multiple retrieval strategies with state-of-the-art Language Models to extract precise answers from complex documents.

This is a readme that focuses on giving an overview of the project. For a more comprehensive look into how the system works, refer to [Explanation.md](./Explanation.md). To just start the app, run ```docker compose build``` and ```docker compose up --build -d```

## Table of Contents

- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Performance Metrics](#performance-metrics)
- [Error Handling](#error-handling)
- [Deployment](#deployment)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Use Cases](#use-cases)
- [Future Roadmap](#future-roadmap)

##  Key Features

### Multi-Format Document Processing
- **PDF, DOCX, PPTX, XLSX, TXT, Markdown, CSV** support
- **Image OCR** with Tesseract (PNG, JPEG, BMP, TIFF, GIF, WebP)
- **Raw text/HTML** fetching from URLs
- **Smart semantic chunking** with overlap for context preservation

### Intelligent Triage Modes
- **Direct Mode**: Small documents (<2000 tokens) processed via Agno agent for ultra-low latency
- **RAG Pipeline**: Large documents with hybrid retrieval and reranking
- **Vision Mode**: Direct image analysis with multimodal LLMs
- **Raw Text Mode**: Web pages and text content processing

### Advanced Retrieval System
- **Hybrid Search**: BM25 (keyword) + FAISS (semantic vector) search
- **Parallel Query Decomposition**: Complex questions broken into sub-queries
- **Reciprocal Rank Fusion (RRF)**: Intelligent result fusion
- **GPU-Accelerated Reranking**: CrossEncoder model for relevance scoring
- **Domain Query Expansion**: Context-aware query enhancement

### Multi-Language & Multi-Model Support
- **LLM Models**: Google Gemini 2.5, OpenAI GPT-4.1, Groq Llama, Groq OpenAI OSS
- **Language Detection**: Automatic fallback for non-English content (<2000 tokens)
- **Batch Processing**: Optimized for multiple questions simultaneously

##  Technical Stack

### Core Technologies
- **Backend**: FastAPI with async processing, asyncio for concurrency
- **LLMs**: Google Gemini 2.5 Flash, OpenAI GPT-4.1, Groq Llama
- **Search**: FAISS (vector similarity), BM25 (keyword matching)
- **Embeddings**: SentenceTransformers Embedding Model (all-MiniLM-L12-v2)
- **Reranking**: CrossEncoder Reranking Model (ms-marco-MiniLM-L-12-v2)

### Document Processing
- **PDF**: PyMuPDF (fitz)
- **Office**: python-docx, python-pptx, pandas(excel)
- **OCR**: Tesseract via pytesseract
- **Download**: aria2c for faster downloads with aiohttp fallback

### Optimization Features
- **GPU Acceleration**: CUDA support for embeddings and FAISS
- **Pre-Emptive Query Generation**: Concurrent pre emptive query expansion for maximum speed
- **Batch Processing**: Concurrent operations with semaphores
- **Caching**: Document and embedding caches
- **Memory Management**: Streaming processing for large files



## Project Structure

```
api/
├── core/
│   ├── agent_logic.py          # Main orchestration logic
│   ├── agno_direct_agent.py    # Direct processing for small docs
│   ├── document_processor.py   # Document parsing and chunking
│   ├── embedding_manager.py    # GPU-optimized embeddings
│   ├── vector_store.py         # Hybrid search implementation
│   └── query_expander.py       # Query enhancement and expansion
├── routes/
│   └── hackrx.py               # API endpoints
├── main.py                     # FastAPI application
├── state.py                    # Global state management
└── settings.py                 # Configuration
```



## Configuration

### Performance Tuning
```python
# GPU Optimization
BATCH_SIZE = 64  # For RTX 4060+
USE_FP16 = True  # Half precision for speed

# Retrieval Parameters
MAX_CHUNKS = 12  # High-K mode
RERANK_CANDIDATES = 25
SIMILARITY_THRESHOLD = 0.3

# Concurrency Limits
QUERY_STRATEGY_SEMAPHORE = 20
ANSWER_SEMAPHORE = 20
SEARCH_SEMAPHORE = 40
```

### Model Selection
- **Small docs**: Groq Llama-3.1-8B (fast inference)
- **Large docs**: Gemini 2.5 Flash (balanced)
- **High accuracy**: GPT-4.1 Mini (premium)

## Performance Metrics

### Processing Speed
- **Small docs (<2K tokens)**: ~1-2 seconds
- **Medium docs (2K-50K tokens)**: ~5-15 seconds  
- **Large docs (50K+ tokens)**: ~10-20 seconds

### Accuracy Features
- **Source Attribution**: Page numbers, clauses, sections and text snippets
- **Confidence Scoring**: Relevance-based ranking
- **Hallucination Prevention**: Evidence-only responses


## Error Handling

### Robust Fallbacks
- **Download failures**: Retry with different methods
- **LLM timeouts**: Automatic fallback to alternative models
- **Parsing errors**: Graceful degradation with partial results
- **GPU unavailable**: CPU fallback for all operations

### Input Validation
- File type verification
- Token limit checking
- URL format validation
- Authentication verification

## Deployment
### Environment Configuration
Create a `.env` based on the .env.sample file:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Docker (Recommended)

Ensure have cloned the MAIN branch
```dockerfile
docker compose up --build -d
```

### Production Considerations
- **GPU Memory**: Monitor VRAM usage for concurrent requests



## How it Works
### Architecture Overview

The system employs a **multi-tier architecture** optimized for latency based on document characteristics:

```
                    Document URL Input
                           ↓
                 ┌─────────────────────┐
                 │   Smart Routing     │
                 │   & Classification  │
                 └─────────┬───────────┘
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
    [Raw Text/HTML]   [Image Files]   [Document Files]
          ↓                ↓                ↓
    Direct LLM Query  Vision Analysis  Intelligence Pipeline
     (~1-2 seconds)   (~2-3 seconds)    (~5-20 seconds)
```

## 📋 Input Classification & Smart Routing

**URL Analysis**: System examines file extensions to route appropriately - no extension/HTML goes to direct processing, image extensions (.png, .jpg, .gif) trigger vision models, document formats (.pdf, .docx, .pptx) enter full RAG pipeline.

**Language Detection**: Samples first 1000 characters to detect language. If document or questions are non-English, bypasses RAG and uses direct multilingual synthesis (for docs <2000 tokens) since embeddings work best in consistent languages.

## 🚀 Processing Modes

### Mode A: Raw Text/HTML (~1-2 seconds)
Fastest mode for web pages/plain text. Direct HTTP fetch → immediate LLM processing with Groq Llama 8B. No chunking, embeddings, or retrieval overhead.

### Mode B: Vision Processing (~2-3 seconds)  
For images, passes URL directly to Groq Llama-4-Scout-17B vision model. Analyzes charts, diagrams, scanned documents without OCR intermediate step.

### Mode C: Direct Document Processing (~1-3 seconds)
For substantial documents under 2000 tokens. Uses Agno agent framework with GPT-OSS-20B, processing full document in single context window without chunking/retrieval.

### Mode D: Full RAG Pipeline (~5-45 seconds)
Most sophisticated mode for large, complex documents. Multi-stage approach preserving semantic relationships while enabling targeted retrieval.

## 🏗️ Knowledge Base Construction

**Intelligent Chunking**: Document-type-aware semantic chunking. Word docs with few line breaks use sentence-based chunking (NLTK), others use paragraph-based with 4096 character limit. Intelligent overlap preserves context across boundaries.

**GPU-Optimized Embeddings**: Uses all-MiniLM-L12-v2 model with batch processing (64 chunks), FP16 precision for 2x speedup. Processes in 256-chunk batches to avoid memory overflow.

**Hybrid Indexing**: Builds FAISS vector index (L2 normalized, GPU-accelerated) and BM25 keyword index concurrently using asyncio.gather(). Provides both semantic similarity and exact term matching.

## 🎯 Query Decomposition & Strategy

**Intelligent Analysis**: Uses GPT-4.1-nano to decompose complex questions into focused sub-queries. Example: "I renewed yesterday, 6 years customer, can I claim Hydrocele?" becomes separate queries about waiting periods and continuous coverage benefits.

**Domain Expansion**: Detects hypotheticals, expands with relevant terminology, identifies acronyms/technical terms using NER and TF-IDF analysis. Creates bidirectional mappings for comprehensive search.

**Concurrent Processing**: All query strategies generated in parallel with 6-second timeout protection and JSON validation fallbacks.

## 🔍 Pre-emptive Search & Retrieval

**Batch Orchestration**: Flattens all sub-queries from all questions into single parallel execution. Simultaneous BM25 and FAISS searches with 40%/60% weighted fusion after score normalization.

**Reciprocal Rank Fusion**: Combines multiple sub-query results using rank-based scoring (1/(k+rank) where k=60). Chunks appearing highly in multiple searches score highest, avoiding incomparable score fusion issues.

**Batched GPU Reranking**: Critical optimization - all (query, chunk) pairs from all questions processed in single CrossEncoder inference call (ms-marco-MiniLM-L-12-v2). Provides ~10x speedup over individual reranking.

## ⚡ Response Generation

**Adaptive Context**: High-K mode (≤18 questions): top 12 chunks + GPT-4.1-mini for accuracy. Fast mode (>18 questions): top 8 chunks + Gemini-2.5-flash-lite for speed.

**Evidence-Only Synthesis**: Explicit prompt constraints prevent hallucination, ignore embedded instructions (prompt injection protection), enforce plain text format. All synthesis happens concurrently with timeout/fallback protection.

**Model Selection Strategy**: 
- Gemini 2.5 Flash Lite: Speed (large batches, non-English)
- GPT-4.1 Mini: Accuracy (small batches, complex questions)  
- Groq Llama: Ultra-fast (direct modes)

## Key Performance Innovations

1. **Concurrent Processing**: Document processing, query analysis, search execution, and synthesis all parallelized
2. **Smart Routing**: Content-aware processing paths avoid unnecessary overhead
3. **Batched Operations**: GPU embedding generation and reranking batched across all questions
4. **Adaptive Strategies**: Dynamic model/context selection based on request characteristics
5. **Timeout Protection**: 6-second hard caps prevent API stalls, graceful fallbacks ensure responses

##  Installation 

(If not using Docker Compose, follow these steps)

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for acceleration)
- Torch with GPU support
- aria2c in path (optional, for faster downloads)

### Setup 
```bash
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### Environment Configuration
Create a `.env` based on the .env.sample file:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```
### Optional : Lint and Format

Feel free to run the ruff scripts in scripts/ to lint and format the code if you want to keep things clean and consistent. We've set up some basic ruff based linting and formatting scripts to catch common issues and standardize formatting.

##  Usage

### Start the API Server

If not using Docker, run:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoint
```
POST /api/v1/hackrx/run
Authorization: Bearer 7bf4409966a1479a8578f3258eba4e215cef0f7ccd694a2440149c1eeb4874ef
Content-Type: application/json
```

### Request Format
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What are the key provisions of Article 21?",
    "What is the waiting period for pre-existing conditions?",
    "Who are the authorized signatories?"
  ]
}
```

### Response Format
```json
{
  "answers": [
    "Article 21 of the Indian Constitution protects the right to life and personal liberty, stating that no person shall be deprived of life or personal liberty except according to procedure established by law.",
    "The waiting period for pre-existing conditions is 24 months as per clause 4.2 of the policy terms.",
    "The authorized signatories are John Smith (CEO) and Mary Johnson (CFO) as specified in section 3.1."
  ]
}
```

## Use Cases

### Legal Documents
- Constitutional articles analysis
- Contract clause extraction
- Legal precedent search

### Insurance Policies
- Coverage details and exclusions
- Claim procedures and waiting periods
- Premium calculations

### Corporate Documents
- Policy interpretation
- Compliance requirements
- Procedure documentation

### Research Papers
- Key findings extraction
- Methodology analysis
- Citation tracking


## Future Roadmap

This roadmap outlines potential directions for evolving the system from a powerful query engine into a comprehensive document intelligence platform.

### Core Pipeline & Performance Enhancements
-   **Graph RAG (Knowledge Graph Retrieval)**: Extract entities and relationships from documents to build a knowledge graph. This would enable answering complex, multi-hop questions like "Which signatories approved contracts with clauses that were later disputed?"
-   **Advanced Table & Chart Understanding**: Implement a dedicated pipeline to parse tables and charts into structured data (e.g., pandas DataFrames). Allow natural language queries directly against this structured data for precise numerical and relational answers.
-   **Hierarchical & Recursive Chunking**: For very long, structured documents (like books or legal codes), implement a strategy that chunks chapters, sections, and paragraphs hierarchically. This allows the system to summarize context at different levels of granularity.
-   **Adaptive Retrieval Strategy**: Develop a meta-agent that learns to select the best retrieval strategy (e.g., BM25-dominant, FAISS-dominant, or specific fusion weights) based on the type of query, improving efficiency and relevance.

### User Experience & Interface
-   **Interactive Document Viewer**: A web UI that displays the source document alongside the chat interface. When an answer is provided, the source chunks should be automatically highlighted in the document viewer for seamless verification.
-   **User Feedback Loop (RLHF)**: Add thumbs-up/thumbs-down buttons to answers. This feedback would be collected to create a dataset for fine-tuning models using Reinforcement Learning from Human Feedback, continuously improving accuracy.
-   **Automated Query Suggestions**: Upon document ingestion, analyze the content and proactively suggest relevant, insightful questions to the user, guiding them toward the document's key information.
-   **Answer Curation & Management**: Allow expert users or administrators to review, edit, and save "golden" answers to common questions, creating a trusted knowledge base that complements the AI-generated responses.

### Enterprise & Production Features
-   **Automated Document Ingestion**: Create data connectors to automatically sync and index documents from sources like **SharePoint, Google Drive, Confluence, and S3 buckets**, keeping the knowledge base up-to-date without manual uploads.
-   **Granular Access Control (ACLs)**: Implement a robust permission system where access can be controlled per document, per collection, or per user group, ensuring data security and confidentiality in a multi-tenant environment.
-   **Comprehensive Analytics Dashboard**: Move beyond logs to a visual dashboard showing query trends, most-queried documents, unanswered questions, latency metrics, and cost-per-query breakdowns for operational insights.
-   **Data Residency & Compliance Controls**: For regulations like GDPR, add features to control the geographical location of data processing and storage, ensuring compliance with international data laws.

### Advanced AI Capabilities
-   **Agentic Workflows & Task Execution**: Evolve from a Q&A system to a task-execution engine. Allow multi-step prompts like, "Summarize the key risks in this contract, compare them against our standard compliance checklist, and draft an email to the legal team highlighting any deviations."
-   **Proactive Insights & Anomaly Detection**: Create a monitoring agent that periodically scans new documents and proactively alerts users to significant changes, risks, or opportunities—for example, "A new termination clause was added to the employee handbook that differs from the previous version."
-   **Multi-Document Comparative Analysis**: Enable queries that span multiple documents, such as "Compare the liability clauses in Contract A and Contract B and highlight the key differences."
-   **Voice-to-Query & Spoken Answers**: Integrate Speech-to-Text (STT) and Text-to-Speech (TTS) services to allow users to ask questions verbally and receive spoken responses, enabling hands-free interaction.