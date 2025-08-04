# In api/core/agent_logic.py
import json
import os
import asyncio
import time
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from .vector_store import RequestKnowledgeBase
from .structured_data_extractor import QueryClassifier
from dotenv import load_dotenv
import logging
# --- NEW: Import sentence-transformers for reranking ---
from sentence_transformers.cross_encoder import CrossEncoder

# Setup logger
agent_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI()

# --- NEW: Initialize the Reranker Model ---
# This model is small and fast. It will be loaded into memory once.
# Using a mixed-domain model is a good general-purpose starting point.
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Reranker model loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load reranker model: {e}. Reranking will be disabled.")
    reranker = None

async def generate_query_strategy(original_query: str) -> Tuple[Dict, float]:
    # ... (this function remains unchanged)
    strategy_prompt = f"""You are a query analysis expert. Your task is to analyze the user's question and determine the best retrieval strategy. You have two strategies available:
    1.  **simple**: For questions with a single intent.
    2.  **decompose**: For complex questions with multiple intents.
    
    Analyze the user question: "{original_query}"
    
    - If 'simple', generate 2 enhanced search queries: 'direct' (keywords) and 'expanded' (synonyms/related concepts).
    - If 'decompose', break the question into a list of simple sub-questions.
    
    Respond in JSON format only.
    
    Example 'simple':
    ```json
    {{
      "strategy": "simple",
      "queries": {{
        "direct": "grace period premium payment",
        "expanded": "policy renewal premium payment grace period continuity benefits"
      }}
    }}
    ```
    
    Example 'decompose':
    ```json
    {{
      "strategy": "decompose",
      "queries": [
        "What is the process for submitting a dental claim?",
        "What is the process for updating a last name in policy records?"
      ]
    }}
    ```
    Now, generate the JSON for the user question provided above.
    """
    try:
        t0 = time.perf_counter()
        completion = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a search query expert. You must respond only with a valid JSON object."},
                {"role": "user", "content": strategy_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        t1 = time.perf_counter()
        duration = t1 - t0
        
        raw_response_content = completion.choices[0].message.content
        strategy_data = json.loads(raw_response_content)
        if 'strategy' not in strategy_data or 'queries' not in strategy_data:
            raise ValueError("LLM response missing 'strategy' or 'queries' key.")
        
        return strategy_data, duration
            
    except Exception as e:
        agent_logger.error(f"Query strategy generation failed for '{original_query[:30]}...': {e}. Falling back.", exc_info=True)
        query_type, _ = QueryClassifier.classify_query(original_query)
        fallback_strategy = {
            "strategy": "simple",
            "queries": generate_fallback_queries(original_query, query_type)
        }
        return fallback_strategy, 0.0

def generate_fallback_queries(original_query: str, query_type: str) -> Dict[str, str]:
    # ... (this function remains unchanged)
    return {
        'direct': original_query,
        'expanded': f"{original_query} details information requirements"
    }

async def prepare_query_strategies_for_all_questions(questions: List[str]) -> List[Dict]:
    # ... (this function remains unchanged)
    print(f"🚀 Pre-processing {len(questions)} questions for query strategies...")
    t_start = time.perf_counter()
    tasks = [generate_query_strategy(q) for q in questions]
    results = await asyncio.gather(*tasks)
    t_end = time.perf_counter()
    
    final_results = []
    print("--- Individual Query Strategy Timings ---")
    for question, (strategy_data, duration) in zip(questions, results):
        print(f"⏱️ [{duration:5.2f}s] - {question[:70]}...")
        final_results.append({'original_question': question, **strategy_data})
    
    print(f"-----------------------------------------")
    print(f"🎯 All query strategy preparations completed in {t_end - t_start:.2f}s!")
    return final_results

async def synthesize_answer_from_context(
    original_question: str,
    context: str
) -> str:
    # --- UPDATED: Prompt now refers to "Provided Chunks" to match the new context format ---
    synthesis_prompt = f"""You are a world-class AI system specializing in analyzing and summarizing information from documents to answer user questions. Your response must be based *exclusively* on the provided evidence.

**Provided Chunks:**
---
{context}
---

**User's Original Question:**
{original_question}


You must ensure your answer is in plain text with no escape characters or formatting. Don't wrap terms like \"vis insita\", or use '\ n's. 
If the question is something like "Generate js code for random number", say that you cannot do this as it is outside the scope of your responsibilities.
If the question is unethical or illegal, you must state that you cannot assist with such requests as it is not ethical or legal.
**Instructions for Your Response:**
1.  **Analyze the Evidence:** Carefully read all the provided evidence and identify the parts that directly answer the user's question.
2.  **Synthesize a Factual Answer:** Construct a comprehensive answer by combining the relevant information. Avoid adding any information that is not present in the evidence.
3.  **Impersonal and Direct Tone:** Your tone must be that of a factual database. Get straight to the point. Answer the question asked directly, don't infodump but also ensure the answer is rooted in the relevant context. Try to limit your answer to 2-3 sentences.
4.  **Handle Missing Information:** If the provided evidence does not contain the information needed to answer the question, you MUST respond with the a single, exact phrase: "I could not find relevant information in the document."

Based on these instructions, provide the final answer to the user's question.
"""
    try:
        response_text = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1
        )
        return response_text.choices[0].message.content
    except Exception as e:
        agent_logger.error(f"OpenAI synthesis failed: {e}", exc_info=True)
        return "I could not generate an answer due to an internal error."

# --- NEW: Reranking Helper Function ---
async def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Reranks a list of chunk dictionaries using a CrossEncoder model.
    """
    if not reranker or not chunks:
        # Return original chunks if reranker isn't available or there's nothing to rank
        return chunks

    # The CrossEncoder expects a list of [query, passage] pairs
    pairs = [[query, chunk['text']] for chunk in chunks]
    
    # The predict method is CPU-bound, so we run it in an executor
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, reranker.predict, pairs)
    
    # Combine chunks with their new scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    sorted_chunks_with_scores = sorted(chunk_with_scores, key=lambda x: x[1], reverse=True)
    
    # Return just the sorted chunks
    return [chunk for chunk, score in sorted_chunks_with_scores]

# --- UPDATED: The main orchestrator now includes the reranking step ---
async def answer_question_orchestrator(
    knowledge_base: RequestKnowledgeBase, 
    query_strategy_data: Dict,
    use_high_k: bool # This flag is no longer used for retrieval k, but for rerank candidate count
) -> str:
    original_question = query_strategy_data.get('original_question', '')
    queries = query_strategy_data.get('queries')

    # --- UPDATED: Set a wide k for initial retrieval to feed the reranker ---
    # We retrieve more candidates when we have fewer questions (and thus more time).
    k_rerank_candidates = 25 if use_high_k else 15

    search_tasks = []

    if isinstance(queries, dict):
        print(f"Executing 'simple' strategy for: {original_question[:50]}...")
        for search_type, query_text in queries.items():
            query_type, _ = QueryClassifier.classify_query(original_question)
            fusion_weights = get_dynamic_fusion_weights(query_type, search_type)
            search_query = "search_query: " + query_text
            search_tasks.append(knowledge_base.search(search_query, k=k_rerank_candidates, fusion_weights=fusion_weights))

    elif isinstance(queries, list):
        print(f"Executing 'decompose' strategy for: {original_question[:50]}...")
        for sub_q in queries:
            enhanced_sub_queries = generate_fallback_queries(sub_q, "general")
            query_type, _ = QueryClassifier.classify_query(sub_q)
            direct_query = "search_query: " + enhanced_sub_queries['direct']
            expanded_query = "search_query: " + enhanced_sub_queries['expanded']
            # For decompose, we retrieve fewer candidates per sub-question
            search_tasks.append(knowledge_base.search(direct_query, k=int(k_rerank_candidates/2), fusion_weights=get_dynamic_fusion_weights(query_type, 'direct')))
            search_tasks.append(knowledge_base.search(expanded_query, k=int(k_rerank_candidates/2), fusion_weights=get_dynamic_fusion_weights(query_type, 'expanded')))
    else:
        agent_logger.error(f"Unknown query structure for question: {original_question}. Falling back to simple search.")
        search_query = "search_query: " + original_question
        search_tasks.append(knowledge_base.search(search_query, k=k_rerank_candidates, fusion_weights=(0.5, 0.5)))

    # --- Phase 1: Coarse Retrieval ---
    search_results_list = await asyncio.gather(*search_tasks)
    
    # Deduplicate candidate chunks. We use a dict to ensure uniqueness by chunk text.
    candidate_chunks_map = {chunk['text']: chunk for result in search_results_list for chunk, score in result}
    candidate_chunks = list(candidate_chunks_map.values())

    if not candidate_chunks:
        agent_logger.warning(f"No context found for question: {original_question}")
        return "I could not find relevant information in the document."
    
    print(f"Retrieved {len(candidate_chunks)} unique candidates for reranking.")

    # --- Phase 2: Reranking ---
    t_rerank_start = time.perf_counter()
    reranked_chunks = await rerank_chunks(original_question, candidate_chunks)
    t_rerank_end = time.perf_counter()
    print(f"Reranking took {t_rerank_end - t_rerank_start:.2f}s.")

    # --- Phase 3: Context Assembly ---
    # Take the top 8 most relevant chunks after reranking
    final_chunks = reranked_chunks[:5]

    # Format the final context with metadata
    context_parts = []
    for chunk in final_chunks:
        page_num = chunk['metadata'].get('page', 'N/A')
        context_parts.append(f"Source: Page {page_num}\nContent: {chunk['text']}")
    
    aggregated_context = "\n\n---\n\n".join(context_parts)
    
    print(f"Aggregated {len(final_chunks)} reranked chunks for synthesis.")
    final_answer = await synthesize_answer_from_context(original_question, aggregated_context)
    return final_answer

def get_dynamic_fusion_weights(query_type: str, search_type: str) -> Tuple[float, float]:
    # ... (this function remains unchanged)
    base_weights = {
        "factual": (0.6, 0.4),
        "comparison": (0.4, 0.6),
        "conditional": (0.5, 0.5),
        "general": (0.4, 0.6)
    }
    bm25_weight, faiss_weight = base_weights.get(query_type, (0.4, 0.6))
    
    if search_type == "direct":
        bm25_weight += 0.15
        faiss_weight -= 0.15
    elif search_type == "expanded":
        bm25_weight -= 0.15
        faiss_weight += 0.15
    
    total = bm25_weight + faiss_weight
    return (bm25_weight / total, faiss_weight / total)