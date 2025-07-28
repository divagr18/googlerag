from typing import Optional
from search_tool import search_insurance_clauses
from agno.agent import Agent,RunResponse
from agno.knowledge.pdf import PDFKnowledgeBase,PDFReader
import environ
env = environ.Env()
env.read_env()
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.document.chunking.document import DocumentChunking
import time
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from agno.utils.pprint import pprint_run_response

class ClauseMapping(BaseModel):
    clause: str = Field(..., description="The clause number or reference used for the decision.")
    explanation: str = Field(..., description="Explanation of how this clause influenced the decision.")

class DecisionResponse(BaseModel):
    decision: str = Field(..., description="Decision outcome, e.g., 'approved' or 'rejected'.")
    amount: Optional[float] = Field(None, description="Applicable amount if decision involves a monetary value.")
    justification: List[str] = Field(..., description="List of justifications, each mapping the decision to specific clause(s).")

from agno.vectordb.milvus import Milvus



api_key = env("QDRANT_API_KEY")
qdrant_url = env("QDRANT_URL")
gmkey = env("GOOGLE_API_KEY")
collection_name = "bajaj"
print("Starting vector db")
vector_db = Milvus(
    collection="bajaj",
    uri="http://localhost:19530",
    search_type="hybrid",
)
print("Starting knowledge base")
knowledge_base = PDFKnowledgeBase(
    vector_db=vector_db, path="bajaj",reader=PDFReader(chunk=True,chunking_strategy=DocumentChunking()))
print("Loading knowledge base")
#knowledge_base.load(recreate=True) 
 # Comment out after first run
agent = Agent(
    model=Gemini(id="gemini-2.5-flash-lite",api_key=gmkey),
    instructions="""Instructions:
        You will be given a short input describing a case in the following format:

"{age}{gender}, {medical issue}, {location}, {policy duration}"

        Example: "46M, knee surgery, Pune, 3-month policy"

        Restrict yourself to max 1 search per run.

        Your task is to respond with the following fields:

        Decision: "approved" or "rejected"

        Amount: Only if applicable. Omit if not relevant.

        Justification: A list of clause-based justifications. Each must include:

        The EXACT clause number or reference

        A clear explanation of how that clause influenced the decision

        Important rules:

        Base your decision on standard health policy rules such as pre-existing condition clauses, waiting periods, age-related exclusions, etc.

        Do not include any free-form text outside the structured fields.

        Every decision must be backed by at least one clause.

        If insufficient data is provided to make a clear judgment, default to "rejected" with an appropriate clause and reasoning.
""",
    debug_mode=True,
    show_tool_calls=True,
    response_model=DecisionResponse,
    knowledge=knowledge_base,
    use_json_mode=True,

    stream_intermediate_steps =True,
    )
print("running agent")
start = time.perf_counter()

agent.run("46M, heart surgery, Pune, 3-month policy", markdown=True)

end = time.perf_counter()
print(f"Agent run completed in {end - start:.3f} seconds")
