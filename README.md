# Document Intelligence & Query System

An advanced document processing and question-answering system that combines multiple retrieval strategies with state-of-the-art Language Models to extract precise answers from complex documents.

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

## 🔧 Technical Stack

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



## 📂 Project Structure

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



## ⚙️ Configuration

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

## 📊 Performance Metrics

### Processing Speed
- **Small docs (<2K tokens)**: ~1-2 seconds
- **Medium docs (2K-50K tokens)**: ~5-15 seconds  
- **Large docs (50K+ tokens)**: ~10-20 seconds

### Accuracy Features
- **Source Attribution**: Page numbers, clauses, sections and text snippets
- **Confidence Scoring**: Relevance-based ranking
- **Hallucination Prevention**: Evidence-only responses


## 🐛 Error Handling

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

## 🚀 Deployment
### Environment Configuration
Create a `.env` based on the .env.sample file:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Docker (Recommended)

Ensure you are on the MAIN branch
```dockerfile
docker compose up --build -d
```

### Production Considerations
- **GPU Memory**: Monitor VRAM usage for concurrent requests



## 🔄 How It Works

### Latency-Aware Architecture Overview

The system employs a **multi-tier architecture** designed for optimal latency based on document characteristics and complexity:

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
     (~1-2 seconds)   (~2-3 seconds)    (~5-45 seconds)
```

### 1. 📋 **Input Classification & Smart Routing**

When a request arrives, the system performs intelligent routing based on the document URL and content characteristics. This decision happens in milliseconds but determines the entire processing strategy.

**URL Analysis**: The system examines the file extension in the URL path. If there's no extension or it appears to be a web page, it's classified as raw text/HTML. Image extensions like `.png`, `.jpg`, `.gif` trigger vision processing. Traditional document formats like `.pdf`, `.docx`, `.pptx` enter the full document intelligence pipeline.

```python
# Classification Logic Flow
if is_raw_text_url(url):           # No file extension or HTML
    → Direct LLM Processing
elif is_image_url(url):            # .png, .jpg, .gif, etc.
    → Vision Model Processing  
else:                              # .pdf, .docx, etc.
    → Document Intelligence Pipeline
```

**Language Detection Strategy**: For document processing, the system samples the first 1000 characters to detect the primary language using the `langdetect` library. It also checks if all questions are in English. This dual-check is crucial because if either the document or questions are non-English, the system bypasses the complex RAG pipeline entirely and uses direct synthesis with multilingual LLMs, which are more capable of handling cross-language scenarios. This path is triggered only if the document is less than 2000 tokens.

```python
# Multi-language handling
doc_sample = full_text[:1000]
doc_en = is_english(doc_sample)
qs_en = all(is_english(q) for q in request.questions)

if not doc_en or not qs_en:
    → Direct synthesis (bypass RAG pipeline)
    → Use multilingual LLM for processing
```

The reasoning here is that embeddings and search work best in consistent languages, so mixed-language scenarios are better handled by powerful multilingual models directly rather than through retrieval systems.

### 2. 🚀 **Processing Modes (Latency Optimized)**

The system employs four distinct processing modes, each optimized for different content types and latency requirements. The key insight is that not all documents need the same level of processing complexity.

#### **Mode A: Raw Text/HTML Processing** (~1-2 seconds)

This is the fastest mode, designed for web pages, API documentation, or plain text content. When the system detects a URL without a file extension or with HTML indicators, it immediately fetches the content using HTTP requests and sends it directly to Groq's Llama model.

The magic here is avoiding any preprocessing - no chunking, no embedding generation, no vector searches. The raw HTML or text content is combined with each question and sent directly to the LLM. This works because modern LLMs can handle substantial context windows, and simple web content doesn't require sophisticated retrieval strategies.

```
URL → HTTP Fetch → Direct LLM Query
                      ↓
                 Groq Llama 8B
                 (Ultra-fast inference)
```

**Data Flow**: The system uses `httpx.AsyncClient` to fetch content, then formats a simple prompt containing the full text and the user's question. The response comes back in 1-2 seconds because there's no intermediate processing.

#### **Mode B: Vision Processing** (~2-3 seconds)

For image inputs, the system recognizes that traditional text processing is irrelevant. Instead of trying to OCR the image and then process the text, it leverages multimodal vision models that can directly understand visual content.

The system passes the image URL directly to Groq's Llama-4-Scout-17B vision model. This model can analyze charts, diagrams, scanned documents, and any visual content without needing intermediate text extraction. The latency is slightly higher than raw text because vision models are computationally more complex, but it's still much faster than OCR followed by text processing.

```
Image URL → Groq Vision Model → Direct Analysis
              ↓
         Llama-4-Scout-17B
         (Multimodal vision)
```

**Why This Works**: Modern vision models have been trained on millions of document images, charts, and diagrams. They can often extract more nuanced information than OCR because they understand visual context, layout, and relationships between elements.

#### **Mode C: Direct Document Processing** (~1-3 seconds)

This mode handles the sweet spot - documents that are substantial enough to require processing but small enough to fit in a single LLM context window. The system uses token counting (via `tiktoken`) to determine if a document is under 2000 tokens.

When this condition is met, the system uses the Agno agent framework, which lets us give the agent tools for agentic url parsing. The Agno agent receives the full document text and processes all questions in sequence without any chunking or retrieval mechanisms.

```
Small Doc → Token Count Check → Agno Agent → Direct LLM
(<2000 tokens)     ↓              ↓
                 Full Text    Context Window
```

**Token Counting Logic**: The system uses OpenAI's `tiktoken` library to count tokens accurately. If that fails, it falls back to a character-based approximation (4 characters per token). This ensures documents that can fit comfortably in modern LLM context windows are processed directly.

**Agno Agent Architecture**: The Agno agent uses Groq's GPT-OSS-20B model for speed while maintaining good quality. It processes each question by combining the full document context with specific instructions, ensuring comprehensive answers without the overhead of retrieval systems.

#### **Mode D: Full RAG Intelligence Pipeline** (~5-45 seconds)

This is the most sophisticated mode, reserved for large, complex documents that exceed the direct processing token limits. The system recognizes that these documents require intelligent segmentation, retrieval, and synthesis strategies.

**The fundamental challenge**: Large documents contain too much information to process at once, but naive chunking loses context and relationships between sections. The RAG pipeline solves this through a multi-stage approach that preserves semantic relationships while enabling targeted retrieval.

**Phase 1: Concurrent Document Processing & Query Preparation**

The RAG pipeline's first innovation is recognizing that document processing and query analysis can happen simultaneously. While the system streams and processes the document, it's also analyzing and decomposing the user's questions in parallel.

```
Document Stream ────┐                    ┌──── Query Strategy
       ↓            │ CONCURRENT TASKS   │        ↓
Text Extraction     │                    │  Query Decomposition
       ↓            │                    │        ↓
Semantic Chunking   │    (Parallelized)  │  Sub-query Generation  
       ↓            │                    │        ↓
Batch Embedding     │                    │  Domain Expansion
       ↓            │                    │        ↓
KB Index Building ──┘                    └──── Strategy Cache
```

**Document Processing Flow**: The system first streams the document using high-speed downloaders like `aria2c`, falling back to `aiohttp` if needed. As chunks of the document arrive, format-specific parsers extract text (PyMuPDF for PDFs, python-docx for Word documents, etc.). This streaming approach means processing begins before the entire document is downloaded.

**Concurrent Query Analysis**: Simultaneously, each user question is sent to GPT-4.1-nano for decomposition analysis. The system asks: "What individual pieces of information are needed to answer this question?" Complex questions like "I renewed my policy yesterday and have been a customer for 6 years. Can I raise a claim for Hydrocele?" get broken down into factual sub-questions about waiting periods and continuous coverage benefits.

This concurrent approach typically saves 3-5 seconds compared to sequential processing, which is crucial for user experience.

### 3. 🏗️ **Knowledge Base Construction**

The knowledge base construction phase is where the system transforms raw document text into a searchable, intelligent repository. This process involves several sophisticated steps that balance context preservation with search efficiency.

#### **Intelligent Chunking Strategy**

Chunking is perhaps the most critical step because it determines how well the system can later retrieve relevant information. The system uses document-type-aware SEMANTIC chunking that adapts to different content structures.

**Word Document Handling**: Microsoft Word documents often contain dense, continuous text with few paragraph breaks. The system detects this by calculating the line break ratio - if there are very few line breaks relative to text length (ratio < 0.001), it switches to sentence-based chunking using NLTK's sentence tokenizer. This ensures that semantic units are preserved even in dense legal or technical documents.

**Paragraph-Based Chunking**: For most other documents, the system splits on double line breaks (paragraph boundaries) but intelligently handles oversized paragraphs by splitting them at the 4096 character limit while preserving sentence boundaries.

**Overlap Strategy**: Critical for maintaining context, the system implements intelligent overlap. When creating new chunks, it includes the last 1-2 sentences from the previous chunk. This ensures that concepts spanning chunk boundaries aren't lost during retrieval.

```python
# Document-type aware chunking
if is_word_doc and line_break_ratio < 0.001:
    → smart_word_doc_chunking()  # Sentence-based for dense text
    → Use NLTK sentence tokenization
    → Overlap preservation for context
else:
    → smart_paragraph_split()    # Paragraph-based
    → MAX_PARA_LENGTH = 4096 characters
```

#### **GPU-Optimized Embedding Pipeline**

Once chunks are created, they need to be converted into vector embeddings for semantic search. This is computationally intensive, so the system is heavily optimized for GPU acceleration.

**Model Selection**: The system uses `all-MiniLM-L12-v2`, which strikes an optimal balance between speed and quality for our RTX 4060 class GPUs. At only 22MB, it loads quickly and processes efficiently.

**Batch Processing**: Instead of processing chunks one by one, the system batches them into groups of 64 (or 16 for CPU). This maximizes GPU utilization and dramatically reduces processing time.

**Precision Optimization**: The system uses FP16 (half-precision) arithmetic, which provides roughly 2x speedup on modern GPUs with minimal quality impact. This is particularly effective for our RTX 4060's Tensor cores.

**Memory Management**: The system processes embeddings in batches of 256 chunks to avoid GPU memory overflow, then concatenates results. This allows processing of arbitrarily large documents without memory constraints.

```
Text Chunks → Batch Processing → GPU Embedding → Vector Storage
    ↓              (64/batch)         ↓              ↓
[chunk1,...]   ────────────→    [vectors]    → FAISS Index
               SentenceTransformer            (normalized L2)
               (FP16 precision)
```

#### **Hybrid Index Construction**

The system builds two complementary search indexes simultaneously to enable hybrid retrieval:

**FAISS Vector Index**: All embeddings are normalized using L2 normalization and stored in a FAISS IndexFlatIP (Inner Product) index. If CUDA is available, the index is moved to GPU for faster search. FAISS provides exact nearest neighbor search for semantic similarity.

**BM25 Keyword Index**: Simultaneously, each chunk is tokenized (extracting words 3+ characters long) and indexed using BM25Okapi. This provides traditional keyword-based search that catches exact term matches that semantic search might miss.

**Parallel Construction**: Both indexes are built concurrently using `asyncio.gather()`, typically completing in 2-3 seconds for most documents:

```python
# Parallel index building
await asyncio.gather(
    _build_bm25_parallel(chunks),      # Keyword search
    _build_faiss_async(embeddings)     # Semantic search
)

# BM25 tokenization (parallel)
tokenized_docs = await asyncio.gather(*[
    loop.run_in_executor(cpu_executor, _tokenize_doc, chunk['text']) 
    for chunk in chunks
])
```

This hybrid approach ensures the system can handle both semantic queries ("what are the benefits?") and specific term searches ("Article 21", "Hydrocele treatment").

### 4. 🎯 **Query Decomposition & Strategy Generation**

Query decomposition is one of the system's most sophisticated features, designed to handle complex, multi-part questions that would otherwise yield poor retrieval results. The system recognizes that human questions often contain multiple information needs that should be searched for separately.

#### **Intelligent Query Analysis**

The decomposition process uses GPT-4.1-nano in a structured reasoning framework. For each question, the system asks the LLM to:

1. **Identify Core Intent**: What is the user fundamentally asking for?
2. **Identify Necessary Facts**: What individual pieces of information are required?
3. **Formulate Sub-Questions**: Create precise, factual queries for each information piece

**Real Example Transformation**: 
```
Original: "I renewed my policy yesterday, been customer for 6 years. Can I claim for Hydrocele?"

Analysis: The personal history is context. The core issue is about waiting periods for a specific condition, potentially modified by continuous coverage.

Sub-Questions:
- "What is the waiting period for Hydrocele treatment?"
- "Are there reductions for continuous coverage from previous years?"
```

This decomposition is crucial because a single complex query often retrieves generic information, while focused sub-queries retrieve the specific facts needed to construct a comprehensive answer.

#### **Domain Query Expansion**

Beyond decomposition, the system employs a sophisticated query expansion system that understands document domains and enhances queries with relevant terminology.

**Hypothetical Question Handling**: The system detects hypothetical questions (containing "if", "suppose", "assuming") and applies domain-specific mappings. For legal questions like "If someone is arrested without warrant, is that legal?", the system expands this to include terms like "arrest", "detention", "authority", "powers", "custody", "warrant", "procedure".

**Technical Term Expansion**: The system builds a domain vocabulary by analyzing the document for:
- **Technical Patterns**: Acronyms, CamelCase terms, hyphenated words
- **Named Entities**: Using NLTK's named entity recognition
- **TF-IDF Analysis**: High-importance terms specific to the document
- **Co-occurrence Relationships**: Terms that frequently appear together

**Acronym Detection and Expansion**: The system automatically detects acronym definitions in text using patterns like "Artificial Intelligence (AI)" and creates bidirectional mappings. When a query mentions "AI", it expands to include "Artificial Intelligence" and vice versa.

**Example Query Expansion Process**:
```
Original Query: "If someone is arrested without warrant, is that legal?"

1. Detect hypothetical: "if someone" indicator found
2. Clean hypothetical framing: "arrested without warrant"
3. Apply legal scenario mapping: ["arrest", "detention", "authority", "powers", "custody", "warrant", "procedure"]
4. Apply general term mapping: "legal" → ["permissible", "authorized", "valid"]
5. Create variants:
   - "arrest detention authority powers custody warrant"
   - "permissible authorized arrest detention warrant procedure"
```

#### **Concurrent Strategy Generation**

The system processes query strategies for all questions simultaneously, dramatically reducing latency:

```python
# Parallel processing for multiple questions
tasks = [generate_query_strategy(q) for q in questions]
results = await asyncio.gather(*tasks)

# Individual timing tracking
for question, (strategy_data, duration) in zip(questions, results):
    print(f"⏱️ [{duration:5.2f}s] - {question[:70]}...")
```

**Timeout Protection**: Each query strategy generation has a 6-second timeout. If the LLM takes too long, the system falls back to using the original question as a single sub-query. This ensures the pipeline never hangs on difficult decompositions.

**JSON Structure Validation**: The system enforces a strict JSON response format from the LLM and validates that required fields exist. If the LLM returns malformed JSON, the system gracefully falls back to the original query.

### 5. 🔍 **Pre-emptive Search & Retrieval**

The retrieval phase is where the system's hybrid architecture truly shines. Rather than performing searches sequentially, the system orchestrates a complex parallel search strategy that minimizes latency while maximizing relevance.

#### **Batch Search Orchestration**

The system recognizes that if you have multiple questions, each potentially decomposed into multiple sub-queries, you could easily have dozens of individual searches to perform. Instead of doing these sequentially, the system flattens all searches into a single batch operation.

**Search Flattening Process**: The system takes all questions, extracts all their sub-queries, and creates one massive list of search operations. For example, if you have 5 questions that decompose into 2-3 sub-queries each, you might have 12-15 total searches. These all execute simultaneously.

```python
# Phase 1: Execute ALL searches in parallel
flattened_search_tasks = []
for sub_questions in all_sub_questions:
    for sub_query in sub_questions:
        flattened_search_tasks.append(
            knowledge_base.search(sub_query, k=k_candidates)
        )

# Single gather for all searches
all_search_results = await asyncio.gather(*flattened_search_tasks)
```

**Hybrid Search Execution**: Each individual search performs both BM25 and FAISS lookups simultaneously, then fuses the results with configurable weights (typically 40% BM25, 60% FAISS). The BM25 search handles exact keyword matches, while FAISS captures semantic similarity.

**Result Normalization**: Because BM25 and FAISS return scores on different scales, the system normalizes both to 0-1 ranges before fusion. This ensures neither search method dominates inappropriately.

#### **Reciprocal Rank Fusion (RRF)**

After individual searches complete, the system faces a new challenge: how to combine results from multiple sub-queries back into a single ranked list for each original question.

**The RRF Algorithm**: RRF assigns scores based on rank position rather than raw similarity scores. For each chunk that appears in search results, its final score is the sum of `1/(k + rank)` across all searches where it appeared. The default k=60 provides a good balance between rewarding high ranks and allowing lower-ranked items to contribute.

**Why RRF Works**: Traditional score fusion often fails because different search algorithms produce incomparable scores. RRF sidesteps this by using only rank information, which is always meaningful. A chunk that appears in the top 3 results for multiple sub-queries will score much higher than one that appears only once.

```python
# Combine multiple sub-query results
def _reciprocal_rank_fusion(search_results_lists, k=60):
    fused_scores = defaultdict(float)
    
    for results_list in search_results_lists:
        for rank, (chunk, score) in enumerate(results_list):
            fused_scores[chunk['text']] += 1 / (k + rank)
    
    return sorted_chunks_by_fused_score
```

**Example**: If a chunk about "Hydrocele waiting periods" appears as rank 1 in one search and rank 3 in another, its RRF score would be `1/(60+0) + 1/(60+2) = 0.0167 + 0.0161 = 0.0328`. This would likely outrank chunks that appear only once, even at rank 1.

#### **Batched GPU Reranking**

The RRF process typically leaves you with 20-30 candidate chunks per question. The final step is reranking these candidates using a more sophisticated model that can assess true relevance.

**The Reranking Challenge**: Reranking models (CrossEncoders) are much more accurate than initial retrieval but also much slower. They need to process every (query, chunk) pair individually. For multiple questions, this could mean hundreds of model calls.

**Batching Innovation**: The system's key optimization is batching ALL reranking operations across ALL questions into a single GPU inference call. Instead of separate reranking for each question, it creates one massive batch of (query, chunk) pairs.

```python
# CRITICAL: Single batch reranking for ALL queries
all_pairs = []  # Flatten all (query, chunk) pairs
for query, chunks in queries_and_chunks:
    pairs = [[query, chunk['text']] for chunk in chunks]
    all_pairs.extend(pairs)

# Single GPU inference call
all_scores = reranker.predict(all_pairs, batch_size=64)
# Split results back to individual queries
```

**Performance Impact**: This batching approach provides roughly 10x speedup over individual reranking. Instead of 10 separate GPU calls for 10 questions, there's one large call that processes 200-300 pairs simultaneously.

**Model Details**: The system uses `CrossEncoder ms-marco-MiniLM-L-12-v2`, which was specifically trained on Microsoft's passage ranking dataset. It's optimized for exactly this use case - determining if a text passage is relevant to a query.

### 6. ⚡ **Latency-Aware Response Generation**

The final synthesis phase demonstrates the system's adaptive intelligence. Rather than using a one-size-fits-all approach, it dynamically adjusts its strategy based on the request characteristics to optimize the balance between speed and accuracy.

#### **Adaptive Context Selection**

The system makes intelligent decisions about how much context to provide to the LLM based on the number of questions being processed:

**High-K Mode (≤18 questions)**: When processing fewer questions, the system can afford to be more thorough. It selects the top 12 reranked chunks per question and uses GPT-4.1-mini for synthesis. This provides maximum accuracy but takes longer per question.

**Fast Mode (>18 questions)**: For large batches, the system prioritizes speed. It reduces context to the top 8 chunks and switches to Gemini-2.5-flash-lite, which has faster inference times. This maintains good quality while dramatically improving throughput.

```python
# Dynamic chunk selection based on question count
if use_high_k:  # ≤18 questions
    final_chunks = reranked_chunks[:12]    # More context
    model = "gpt-4.1-mini"                 # Higher accuracy
else:           # >18 questions  
    final_chunks = reranked_chunks[:8]     # Less context
    model = "gemini-2.5-flash-lite"       # Faster inference
```

**Context Assembly**: For each question, the selected chunks are formatted with source page numbers and combined into a structured prompt. The context clearly separates different source chunks while maintaining their provenance information.

#### **Evidence-Only Synthesis**

The synthesis prompt is carefully designed to prevent hallucination and ensure answers are grounded in the provided evidence:

**Security Instructions**: The prompt explicitly warns the LLM to ignore any instructions that might be embedded within the document content. This prevents prompt injection attacks where malicious documents try to override the system's behavior.

**Evidence Constraint**: The LLM is instructed to base responses exclusively on the provided chunks and to explicitly state when information isn't available in the document.

**Format Control**: The system enforces plain text responses without markdown or special formatting, ensuring consistent output that works well in various presentation contexts.

#### **Concurrent Synthesis**

Just as with other phases, synthesis happens in parallel for all questions:

```python
# Parallel LLM calls for all questions
synthesis_tasks = [
    synthesize_answer_from_context(question, context, use_high_k)
    for question, context in zip(questions, contexts)
]

final_answers = await asyncio.gather(*synthesis_tasks)
```

**Model Selection Strategy**: The system uses different models based on context and requirements:
- **Gemini 2.5 Flash Lite**: For speed (large batches, non-English)
- **GPT-4.1 Mini**: For accuracy (small batches, complex questions)
- **Groq Llama**: For ultra-fast processing (direct modes)

**Timeout and Fallback**: Each synthesis call has timeout protection. If the primary model fails, the system automatically retries with a backup model (e.g., Gemini Flash), ensuring a response is always generated.


## 🛠️ Installation (If not using Docker Compose)

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for acceleration)
- Torch with GPU support
- aria2c (optional, for faster downloads)

### Setup
```bash
git clone https://github.com/divagr18/bajajrx.git
cd bajajrx
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


## 🚀 Usage

### Start the API Server
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

## 🎯 Use Cases

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


## 📈 Future Roadmap

### Immediate Enhancements
- [ ] Persistent vector database integration
- [ ] Web UI for document upload and querying
- [ ] Multi-tenant authentication system
- [ ] Real-time processing status updates

### Advanced Features
- [ ] Conversational follow-up questions
- [ ] Document summarization and key insights
- [ ] Multi-document cross-referencing
- [ ] Custom domain model fine-tuning

### Integration Possibilites
- [ ] Microsoft Teams/Slack bots
- [ ] REST API SDK generation
- [ ] Webhook notifications
- [ ] Export to various formats