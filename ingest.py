import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import pypdf

# --- 1. Configuration ---
load_dotenv()
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "bajaj_insurance_pdfs" # New collection name for clarity
EMBEDDING_DIM = 1536
OPENAI_MODEL_NAME = "text-embedding-3-small"
PDF_DIRECTORY = "bajaj" # The folder containing your PDF files

# --- 2. Initialize Clients and Connections ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not openai_client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print("✅ Connected to Milvus")

# --- 3. Helper Functions for PDF Processing ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Splits text into chunks based on paragraphs, then by size."""
    # First, split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            # If a paragraph is too long, split it by character count
            for i in range(0, len(p), chunk_size - overlap):
                chunks.append(p[i:i + chunk_size])
    return chunks

# --- 4. Main Ingestion Logic ---

def run_ingestion():
    """Finds PDFs, chunks them, and inserts them into Milvus."""
    # A. Setup Milvus Collection
    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096), # Increased max_length for larger chunks
        FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="location_covered", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="min_policy_months_for_coverage", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, "Insurance policy PDF collection")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"✅ Collection '{COLLECTION_NAME}' created.")

    # B. Find and Process PDFs
    pdf_paths = glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf"))
    if not pdf_paths:
        print(f"❌ No PDF files found in the '{PDF_DIRECTORY}' folder. Please add some.")
        return

    print(f"Found {len(pdf_paths)} PDF(s) to process...")
    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"  - Processing: {os.path.basename(pdf_path)}")
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            continue
        
        text_chunks = chunk_text(document_text)
        for chunk in text_chunks:
            all_chunks.append({
                "text": chunk,
                "source_document": os.path.basename(pdf_path),
                # Assigning default metadata. The specifics are in the text itself.
                "location_covered": "pune,mumbai,delhi,chennai,bangalore",
                "min_policy_months_for_coverage": 0
            })
    
    print(f"✅ Generated a total of {len(all_chunks)} text chunks from all PDFs.")

    # C. Embed and Insert in Batches
    BATCH_SIZE = 100 # Process 100 chunks at a time to avoid API limits
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        print(f"  - Processing batch {i//BATCH_SIZE + 1}/{(len(all_chunks) + BATCH_SIZE - 1)//BATCH_SIZE}...")
        
        # Get texts for the current batch
        batch_texts = [item['text'] for item in batch]
        
        # Generate embeddings
        response = openai_client.embeddings.create(input=batch_texts, model=OPENAI_MODEL_NAME)
        embeddings = [item.embedding for item in response.data]
        
        # Prepare data for Milvus insertion
        data_to_insert = [
            [item['text'] for item in batch],
            [item['source_document'] for item in batch],
            [item['location_covered'] for item in batch],
            [item['min_policy_months_for_coverage'] for item in batch],
            embeddings
        ]
        
        collection.insert(data_to_insert)

    print("✅ All batches inserted.")
    collection.flush()
    
    # D. Create Index and Load
    print("Creating vector index...")
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="vector", index_params=index_params)
    print("✅ Vector index created.")
    
    collection.load()
    print("✅ Collection loaded into memory.")
    print("\n🎉 PDF ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
    connections.disconnect("default")