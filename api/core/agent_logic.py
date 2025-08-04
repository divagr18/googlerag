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

# Setup logger
agent_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI()

async def generate_query_strategy(original_query: str) -> Tuple[Dict, float]:
    strategy_prompt = f"""You are a query analysis expert. Your task is to analyze the user's question and determine the best retrieval strategy. You have two strategies available:
    1.  **simple**: For questions with a single intent.
    2.  **decompose**: For complex questions with multiple intents.
    
    Analyze the user question: "{original_query}"
    
    - If 'simple', generate 2 enhanced search queries: 'direct' (direct queries) and 'expanded' (synonyms/related concept queries). Try not to overlap the two.
    - If 'decompose', break the question into a list of simple sub-questions.
    
    Respond in JSON format only.
    
    Example 'simple':
    ```json
    {{
    "strategy": "simple",
    "queries": {
        "direct": "What is the grace period for insurance premium payments?",
        "expanded": "What happens if an insurance premium is paid late, and how long does coverage remain active during the grace period?"
    }
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
            model="gpt-4.1-nano",
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
    return {
        'direct': original_query,
        'expanded': f"{original_query} details information requirements"
    }

async def prepare_query_strategies_for_all_questions(questions: List[str]) -> List[Dict]:
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
    synthesis_prompt = f"""You are a world-class AI system specializing in analyzing and summarizing information from documents to answer user questions. Your response must be based *exclusively* on the provided evidence.

**Provided Evidence:**
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
4.  **Handle Missing Information:** If the provided evidence does not contain the information needed to answer the question, you MUST respond with the single, exact phrase: "I could not find relevant information in the document." Do not guess or use external knowledge.
5.  **Be Detailed but Concise:** Provide a complete answer, but do not include unnecessary details or long quotes. Summarize the rules and facts in your own words.

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

# FIX: Orchestrator now accepts the enrichment_enabled flag.
async def answer_question_orchestrator(
    knowledge_base: RequestKnowledgeBase, 
    query_strategy_data: Dict,
    enrichment_enabled: bool
) -> str:
    MIN_CHUNKS = 8
    strategy = query_strategy_data.get('strategy', 'simple')
    original_question = query_strategy_data.get('original_question', '')
    
    print(f"Executing '{strategy}' strategy for: {original_question[:50]}...")

    search_tasks = []
    
    if strategy == 'simple':
        enhanced_queries = query_strategy_data['queries']
        for search_type, query_text in enhanced_queries.items():
            query_type, _ = QueryClassifier.classify_query(original_question)
            fusion_weights = get_dynamic_fusion_weights(query_type, search_type)
            search_tasks.append(knowledge_base.search(f"search_query: {query_text}", k=10, fusion_weights=fusion_weights))

    elif strategy == 'decompose':
        sub_questions = query_strategy_data['queries']
        for sub_q in sub_questions:
            enhanced_sub_queries = generate_fallback_queries(sub_q, "general")
            query_type, _ = QueryClassifier.classify_query(sub_q)
            search_tasks.append(knowledge_base.search(f"search_query: {enhanced_sub_queries['direct']}", k=6, fusion_weights=get_dynamic_fusion_weights(query_type, 'direct')))
            search_tasks.append(knowledge_base.search(f"search_query: {enhanced_sub_queries['expanded']}", k=6, fusion_weights=get_dynamic_fusion_weights(query_type, 'expanded')))

    search_results_list = await asyncio.gather(*search_tasks)
    
    all_chunks = set()
    for result_list in search_results_list:
        for chunk in result_list:
            all_chunks.add(chunk)
            
    # FIX: Context enrichment is now conditional on the flag passed from the endpoint.
    if enrichment_enabled and 0 < len(all_chunks) < MIN_CHUNKS:
        print(f"⚠️ Low chunk count ({len(all_chunks)}). Performing supplemental search to reach {MIN_CHUNKS}...")
        supplemental_chunks = await knowledge_base.search(
            f"search_query: {original_question}", 
            k=10, 
            fusion_weights=(0.5, 0.5)
        )
        all_chunks.update(supplemental_chunks)

    if not all_chunks:
        agent_logger.warning(f"No context found for question: {original_question}")
        return "I could not find relevant information in the document."

    chunk_list = list(all_chunks)
    if len(chunk_list) > 8:
        print(f"Aggregated {len(chunk_list)} unique chunks, capping to 8 for synthesis.")
        final_chunks = chunk_list[:8]
    else:
        print(f"Aggregated {len(chunk_list)} unique chunks for synthesis.")
        final_chunks = chunk_list

    aggregated_context = "\n\n---\n\n".join(final_chunks)
    
    final_answer = await synthesize_answer_from_context(original_question, aggregated_context)
    return final_answer

def get_dynamic_fusion_weights(query_type: str, search_type: str) -> Tuple[float, float]:
    base_weights = {
        "factual": (0.7, 0.3),
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