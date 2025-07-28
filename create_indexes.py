# Filename: create_indexes.py

import qdrant_client
from qdrant_client.http.models import PayloadSchemaType

# --- 1. Configuration: Replace these values ---
QDRANT_URL = "1abc2a6c-3abe-41c8-9224-5b20498462f2.eu-central-1-0.aws.cloud.qdrant.io"  # e.g., "https://xyz-abc.us-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.AWkhKIRwluVtUpbC_S4aGgZfU9LNTll4h1OLmemoV3c"
COLLECTION_NAME = "bajaj" # The name of your existing collection

# --- 2. Define the fields you want to index from your payload ---
# This should match the field names in the JSON payload of your points.
LOCATION_FIELD = "location_covered"
POLICY_MONTHS_FIELD = "min_policy_months_for_coverage"

def setup_payload_indexes():
    """
    Connects to Qdrant and creates payload indexes for faster filtering.
    """
    try:
        # Initialize the Qdrant client
        client = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

        print(f"Successfully connected to Qdrant.")
        print(f"Attempting to create indexes on collection: '{COLLECTION_NAME}'")

        # --- Create Index for the Location Field ---
        # We use a 'keyword' index for fields that store lists of strings or single strings.
        # This is ideal for exact matching (e.g., location = 'pune').
        print(f"Creating payload index for field: '{LOCATION_FIELD}' (Type: Keyword)...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=LOCATION_FIELD,
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"--> Index for '{LOCATION_FIELD}' created or already exists.")

        # --- Create Index for the Policy Months Field ---
        # We use an 'integer' index for numerical fields to allow for efficient
        # range queries (e.g., policy_months >= 3).
        print(f"Creating payload index for field: '{POLICY_MONTHS_FIELD}' (Type: Integer)...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=POLICY_MONTHS_FIELD,
            field_schema=PayloadSchemaType.INTEGER
        )
        print(f"--> Index for '{POLICY_MONTHS_FIELD}' created or already exists.")

        print("\nPayload indexing setup complete!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your QDRANT_URL, QDRANT_API_KEY, and COLLECTION_NAME.")

if __name__ == "__main__":
    setup_payload_indexes()