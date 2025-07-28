import os
import re
import json,time
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Any
from pymilvus import connections, Collection

# --- 1. Configuration ---
load_dotenv()
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "bajaj_insurance_pdfs" # Use the collection with real PDF data
OPENAI_MODEL_NAME = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini" # A fast and capable model for generation

# --- 2. Initialize Clients and Connections ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
collection.load()

# --- 3. Search Function (Unchanged) ---
def search_insurance_clauses(
    query_text: str, location: Optional[str] = None, policy_months: Optional[int] = None, limit: int = 3
) -> List[Dict[str, Any]]:
    """Queries the Milvus server for relevant clauses with optional filters."""
    response = openai_client.embeddings.create(input=[query_text], model=OPENAI_MODEL_NAME)
    query_vector = response.data[0].embedding
    filter_conditions = []
    if location:
        filter_conditions.append(f'location_covered like "%{location.lower()}%"')
    if policy_months is not None:
        filter_conditions.append(f"min_policy_months_for_coverage <= {policy_months}")
    query_expr = " and ".join(filter_conditions) if filter_conditions else ""
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector], anns_field="vector", param=search_params,
        limit=limit, expr=query_expr, output_fields=["text", "source_document"]
    )
    formatted_results = []
    if results:
        for hit in results[0]:
            formatted_results.append({
                "text": hit.entity.get("text"), "source": hit.entity.get("source_document"), "distance": hit.distance
            })
    return formatted_results

# --- 4. NEW: LLM-Powered Justification Generator ---
def generate_explanation_with_llm(query: str, decision: str, clauses: List[Dict[str, Any]]) -> str:
    """
    Uses an LLM to generate a human-readable justification for a decision.
    """
    if not clauses:
        return "No relevant policy information was found to support a decision."

    # Prepare the context for the LLM
    clauses_text = "\n\n".join([f"Clause from '{c['source']}':\n>>> {c['text']}" for c in clauses])
    
    # Construct a precise prompt
    prompt = f"""
    You are an insurance claims analyst. Your task is to write a clear, concise justification for a claim decision.

    Original User Query: "{query}"
    
    Relevant Policy Clauses:
    ---
    {clauses_text}
    ---
    
    A decision has already been made: **{decision}**

    Based *only* on the provided policy clauses, explain in simple terms *why* this decision was reached.
    Directly quote the most critical part of the relevant clause in your explanation.
    Do not invent new information. If the clauses are unclear, state that the decision is based on the most relevant information found.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful insurance claims analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Make the output deterministic
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating LLM explanation: {e}")
        return "Could not generate a detailed explanation due to an error."


# --- 5. The Main Processor Tool (Updated) ---
def process_claim_direct(query: str) -> Dict[str, Any]:
    """
    Processes a raw claim query using a hybrid rule-based and LLM approach.
    """
    print(f"\n--- Processing Query: '{query}' ---")

    # Step A: Parse
    query_lower = query.lower()
    policy_months_match = re.search(r'(\d+)[-\s]?month', query_lower)
    policy_months = int(policy_months_match.group(1)) if policy_months_match else None
    known_locations = ["pune", "mumbai", "delhi", "bangalore", "chennai"]
    location = next((loc for loc in known_locations if loc in query_lower), None)
    procedure_text = re.sub(r'(\d+)[-\s]?month.*policy|' + '|'.join(known_locations) + r'|\d+m|\d+f|\d+[-\s]?y(ear|o)?', '', query_lower, flags=re.I).strip(', ')
    
    print(f"Parsed -> Procedure: '{procedure_text}', Location: {location}, Policy Months: {policy_months}")

    # Step B: Search
    retrieved_clauses = search_insurance_clauses(
        query_text=procedure_text, location=location, policy_months=policy_months
    )

    # Step C: Decide (Deterministic Rules)
    decision = "Approved" # Default to approved
    if not retrieved_clauses:
        decision = "Rejected"
    else:
        top_clause = retrieved_clauses[0]
        exclusion_keywords = ["not covered", "exclusion", "does not cover"]
        if any(keyword in top_clause['text'].lower() for keyword in exclusion_keywords):
            decision = "Rejected"
        
        # Example of a more complex rule: Check for waiting periods
        waiting_period_match = re.search(r'(\d+)[-\s]?month waiting period', top_clause['text'].lower())
        if waiting_period_match and policy_months:
            required_months = int(waiting_period_match.group(1))
            if policy_months < required_months:
                decision = "Rejected"

    print(f"Rule-based Decision -> {decision}")

    # Step D: Generate Justification (LLM-powered)
    justification = generate_explanation_with_llm(query, decision, retrieved_clauses)
    
    return {
        "decision": decision,
        "justification": justification,
        "supporting_clauses": retrieved_clauses
    }

# --- 6. Testing Block ---
if __name__ == "__main__":
    start = time.perf_counter()

    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    ]
    all_results = []
    for q in test_queries:
        result = process_claim_direct(q)
        all_results.append({"query": q, "response": result})
        print("-" * 20)
    print("\n\n--- FINAL JSON SUMMARY ---")
    end = time.perf_counter()
    print(f"Agent run completed in {end - start:.3f} seconds")
    print(json.dumps(all_results, indent=2))
    connections.disconnect("default")