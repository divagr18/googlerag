import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Any
from pymilvus import connections, Collection

# --- Configuration ---
load_dotenv()
# Use the new collection name and DB file
COLLECTION_NAME = "bajaj"
MILVUS_FILE = "http://localhost:19530"
OPENAI_MODEL_NAME = "text-embedding-3-small"

# --- Initialize Models and Connection ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not openai_client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

connections.connect("default", uri=MILVUS_FILE)
collection = Collection(COLLECTION_NAME)
collection.load()

def search_insurance_clauses(
    query_text: str,
    location: Optional[str] = None,  # Still accepting location, but won't use it
    policy_months: Optional[int] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Searches for relevant insurance clauses in the Milvus knowledge base using OpenAI embeddings.
    Performs a semantic search on the query_text and can filter by policy age (location filter removed).

    Args:
        query_text (str): The main topic or question to search for.
        location (Optional[str]): [IGNORED] Location filtering has been disabled.
        policy_months (Optional[int]): The age of the user's policy in months.
        limit (int): Max number of results to return.

    Returns:
        List[Dict[str, Any]]: Matching documents.
    """

    print("=" * 120)
    print(f"DEBUG Running: search_insurance_clauses(query_text={query_text}, policy_months={policy_months}, limit={limit})")

    # 1. Embed query
    print("DEBUG Generating embedding for query text...")
    response = openai_client.embeddings.create(input=[query_text], model=OPENAI_MODEL_NAME)
    query_vector = response.data[0].embedding
    print("DEBUG Embedding generated.")

    # 2. Build expression (location filter removed)
    filter_conditions = []
    query_expr = " and ".join(filter_conditions) if filter_conditions else ""
    print(f"DEBUG Filter expression: {query_expr if query_expr else '[No filters applied]'}")

    # 3. Define search parameters
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    print("DEBUG Search parameters:", search_params)

    # 4. Perform search
    print("DEBUG Performing search on collection...")
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=limit,
        expr=query_expr,
        output_fields=["text", "source_document"]
    )
    print("DEBUG Search complete.")

    # 5. Format results
    formatted_results = []
    for hit in results[0]:
        formatted_results.append({
            "text": hit.entity.get("text", "No text available"),
            "source": hit.entity.get("source_document", "N/A"),
            "score": hit.distance,
        })

    print(f"DEBUG Retrieved {len(formatted_results)} results.")
    print("=" * 120)
    print("DEBUG Results:", formatted_results)
    return formatted_results
