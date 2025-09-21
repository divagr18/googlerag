# api/core/guardian_score.py
import re
import json
import asyncio
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
class ExploitationType(Enum):
    FINANCIAL_EXPLOITATION = "financial_exploitation"
    LEGAL_RIGHTS_VIOLATION = "legal_rights_violation"
    UNFAIR_TERMINATION = "unfair_termination"
    EXCESSIVE_PENALTIES = "excessive_penalties"
    MAINTENANCE_BURDEN = "maintenance_burden"
    PRIVACY_VIOLATION = "privacy_violation"
    DISCRIMINATORY_TERMS = "discriminatory_terms"
    # General exploitation types for all contracts
    POWER_IMBALANCE = "power_imbalance"
    UNCONSCIONABLE_TERMS = "unconscionable_terms"
    MODIFICATION_ABUSE = "modification_abuse"
    DISPUTE_RESOLUTION_BIAS = "dispute_resolution_bias"
    INTELLECTUAL_PROPERTY_OVERREACH = "intellectual_property_overreach"
    NON_COMPETE_ABUSE = "non_compete_abuse"
    SALARY_MANIPULATION = "salary_manipulation"
    WORK_CONDITION_VIOLATION = "work_condition_violation"

@dataclass
class ExploitationFlag:
    type: ExploitationType
    risk_level: RiskLevel
    description: str
    clause_text: str
    severity_score: int  # 0-100
    recommendation: str
    ai_recommendation: str = ""  # AI-generated contextual recommendation

@dataclass
class GuardianScoreResult:
    overall_score: int  # 0-100 (100 = completely safe, 0 = extremely exploitative)
    risk_level: RiskLevel
    exploitation_flags: List[ExploitationFlag]
    missing_protections: List[str]
    fair_clauses: List[str]
    summary: str

class GuardianScoreAnalyzer:
    """
    Analyzes contracts against ideal templates to detect exploitation
    and protect users from unfair terms. Uses Gemini API for advanced analysis.
    """
    
    def __init__(self):
        self.exploitation_patterns = self._load_exploitation_patterns()
        self.fair_clause_patterns = self._load_fair_clause_patterns()
        self.required_protections = self._load_required_protections()
    
    async def _llm_based_harmful_clause_detection(self, contract_text: str) -> List[ExploitationFlag]:
        """
        Use Gemini LLM to detect potentially harmful clauses that regex patterns might miss.
        This provides a more nuanced understanding of contract language.
        Processes multiple chunks in parallel for better performance.
        """
        logger.info("🤖 Starting LLM-based harmful clause detection with Gemini API")
        
        try:
            # Import here to avoid circular imports
            from google import genai
            from google.genai import types
            import os
            
            # Initialize Gemini client
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found, skipping LLM-based detection")
                return []
            
            gemini_client = genai.Client(api_key=api_key)
            
            # Split contract into manageable chunks for analysis
            chunks = self._split_contract_into_chunks(contract_text)
            logger.info(f"📄 Split contract into {len(chunks)} chunks for LLM analysis")
            
            # Limit to maximum 30 chunks
            if len(chunks) > 30:
                logger.warning(f"Contract too large ({len(chunks)} chunks), limiting to first 30 chunks")
                chunks = chunks[:30]
            
            async def analyze_chunk(chunk_idx: int, chunk: str) -> List[ExploitationFlag]:
                """Analyze a single chunk with Gemini"""
                logger.info(f"🔍 Analyzing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} characters)")
                
                # Prompt for LLM to analyze harmful clauses
                analysis_prompt = f"""
                Analyze the following contract text for potentially harmful, exploitative, or unconscionable clauses.
                Focus on identifying:
                1. Unfair power imbalances
                2. Excessive penalties or financial burdens
                3. Violation of worker/tenant rights
                4. Unreasonable restrictions or obligations
                5. Clauses that may be illegal or unenforceable
                6. Hidden costs or financial obligations
                7. Waiver of important legal rights
                8. Overly broad non-compete or confidentiality terms
                
                IMPORTANT: For the "clause_text" field, provide the EXACT text from the contract as it appears, including punctuation and formatting. This text will be highlighted in the original document, so it must match exactly.
                
                For each problematic clause you find, provide:
                - The specific problematic text (EXACT quote from contract - minimum 15 characters)
                - Why it's harmful or unfair
                - Severity level (1-100, where 100 is extremely dangerous)
                - Type of exploitation (financial_exploitation, legal_rights_violation, power_imbalance, etc.)
                
                Contract text to analyze:
                {chunk}
                
                Respond in JSON format:
                {{
                    "harmful_clauses": [
                        {{
                            "clause_text": "EXACT quote from contract as it appears",
                            "harm_description": "explanation of why this is harmful",
                            "severity": 85,
                            "exploitation_type": "power_imbalance",
                            "recommendation": "specific advice for addressing this"
                        }}
                    ]
                }}
                """
                
                try:
                    # Get AI analysis using Gemini
                    response = await gemini_client.aio.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=[analysis_prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            system_instruction="You are a legal expert analyzing contracts for exploitation. Return only valid JSON.",
                        ),
                    )
                    
                    logger.info(f"📥 Gemini raw response for chunk {chunk_idx}: {response.text[:500]}...")
                    
                    # Clean and parse AI response
                    raw_text = response.text.strip()
                    
                    # Try to extract JSON from response if it's wrapped in markdown
                    if raw_text.startswith("```json"):
                        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                    elif raw_text.startswith("```"):
                        raw_text = raw_text.replace("```", "").strip()
                    
                    # Try to find JSON object in the response
                    json_start = raw_text.find("{")
                    json_end = raw_text.rfind("}") + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = raw_text[json_start:json_end]
                        logger.info(f"🔧 Extracted JSON for chunk {chunk_idx}: {json_text[:200]}...")
                        try:
                            analysis = json.loads(json_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"❌ JSON parsing failed for chunk {chunk_idx}: {str(e)}")
                            logger.debug(f"Failed JSON text: {json_text}")
                            return []
                    else:
                        logger.warning(f"❌ No JSON object found in Gemini response for chunk {chunk_idx}")
                        logger.debug(f"Raw response: {raw_text}")
                        return []
                    
                    # Convert AI findings to ExploitationFlag objects
                    chunk_flags = []
                    for clause_finding in analysis.get("harmful_clauses", []):
                        try:
                            exploitation_type = self._map_ai_type_to_enum(clause_finding.get("exploitation_type", "power_imbalance"))
                            risk_level = self._determine_risk_level(clause_finding.get("severity", 50))
                            
                            # Get the clause text and ensure it's valid for highlighting
                            raw_clause_text = clause_finding.get("clause_text", "")
                            
                            # Don't truncate too aggressively for LLM clauses - they need to match original text
                            if len(raw_clause_text) > 200:
                                # For LLM clauses, try to find a good sentence break
                                sentence_break = raw_clause_text.find('.', 150)
                                if sentence_break != -1 and sentence_break < 200:
                                    processed_clause_text = raw_clause_text[:sentence_break + 1].strip()
                                else:
                                    # Try word boundary
                                    word_break = raw_clause_text.rfind(' ', 150, 200)
                                    if word_break != -1:
                                        processed_clause_text = raw_clause_text[:word_break].strip() + "..."
                                    else:
                                        processed_clause_text = raw_clause_text[:200].strip() + "..."
                            else:
                                processed_clause_text = raw_clause_text.strip()
                            
                            # Skip if clause text is too short or empty
                            if len(processed_clause_text) < 10:
                                logger.warning(f"⚠️ Skipping LLM flag with too short clause text: '{processed_clause_text}'")
                                continue
                            
                            flag = ExploitationFlag(
                                type=exploitation_type,
                                risk_level=risk_level,
                                description=f"AI-detected: {clause_finding.get('harm_description', 'Potentially harmful clause')}",
                                clause_text=processed_clause_text,
                                severity_score=clause_finding.get("severity", 50),
                                recommendation=clause_finding.get("recommendation", "Review this clause with legal counsel"),
                                ai_recommendation=clause_finding.get("recommendation", "")
                            )
                            chunk_flags.append(flag)
                            logger.info(f"✅ Added LLM flag from chunk {chunk_idx}: {exploitation_type.value} (severity: {flag.severity_score})")
                            logger.debug(f"📝 LLM clause text: '{processed_clause_text[:100]}...'")
                        except Exception as e:
                            logger.warning(f"❌ Error processing AI clause finding from chunk {chunk_idx}: {e}")
                            continue
                    
                    return chunk_flags
                            
                except json.JSONDecodeError:
                    logger.warning(f"❌ Gemini returned invalid JSON for chunk {chunk_idx}")
                    return []
                except Exception as e:
                    logger.warning(f"❌ Gemini analysis failed for chunk {chunk_idx}: {str(e)}")
                    return []
            
            # Process chunks in parallel batches of up to 10
            batch_size = 10
            all_flags = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                logger.info(f"🚀 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} with {len(batch_chunks)} chunks")
                
                # Create tasks for parallel processing
                tasks = [
                    analyze_chunk(i + j, chunk) 
                    for j, chunk in enumerate(batch_chunks)
                ]
                
                # Execute batch in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results from this batch
                for result in batch_results:
                    if isinstance(result, list):
                        all_flags.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"❌ Chunk analysis failed with exception: {result}")
            
            logger.info(f"🎯 Gemini LLM detected {len(all_flags)} additional harmful clauses across {len(chunks)} chunks")
            return all_flags
            
        except Exception as e:
            logger.error(f"❌ LLM-based analysis failed: {e}")
            return []
    
    def _split_contract_into_chunks(self, contract_text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split contract into chunks for LLM analysis"""
        # Split by paragraphs or sections first
        paragraphs = contract_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _map_ai_type_to_enum(self, ai_type: str) -> ExploitationType:
        """Map AI-generated exploitation type to our enum"""
        type_mapping = {
            "financial": ExploitationType.FINANCIAL_EXPLOITATION,
            "financial_exploitation": ExploitationType.FINANCIAL_EXPLOITATION,
            "legal_rights": ExploitationType.LEGAL_RIGHTS_VIOLATION,
            "legal_rights_violation": ExploitationType.LEGAL_RIGHTS_VIOLATION,
            "power_imbalance": ExploitationType.POWER_IMBALANCE,
            "unconscionable": ExploitationType.UNCONSCIONABLE_TERMS,
            "unconscionable_terms": ExploitationType.UNCONSCIONABLE_TERMS,
            "modification": ExploitationType.MODIFICATION_ABUSE,
            "modification_abuse": ExploitationType.MODIFICATION_ABUSE,
            "dispute_resolution": ExploitationType.DISPUTE_RESOLUTION_BIAS,
            "dispute_resolution_bias": ExploitationType.DISPUTE_RESOLUTION_BIAS,
            "intellectual_property": ExploitationType.INTELLECTUAL_PROPERTY_OVERREACH,
            "ip_overreach": ExploitationType.INTELLECTUAL_PROPERTY_OVERREACH,
            "non_compete": ExploitationType.NON_COMPETE_ABUSE,
            "non_compete_abuse": ExploitationType.NON_COMPETE_ABUSE,
            "salary": ExploitationType.SALARY_MANIPULATION,
            "salary_manipulation": ExploitationType.SALARY_MANIPULATION,
            "work_conditions": ExploitationType.WORK_CONDITION_VIOLATION,
            "work_condition_violation": ExploitationType.WORK_CONDITION_VIOLATION,
            "termination": ExploitationType.UNFAIR_TERMINATION,
            "unfair_termination": ExploitationType.UNFAIR_TERMINATION,
            "privacy": ExploitationType.PRIVACY_VIOLATION,
            "privacy_violation": ExploitationType.PRIVACY_VIOLATION,
        }
        return type_mapping.get(ai_type.lower(), ExploitationType.POWER_IMBALANCE)
    
    def _safe_truncate_clause_text(self, text: str, max_length: int = 120) -> str:
        """
        Safely truncate clause text for frontend display and regex safety.
        Finds good break points to avoid cutting words in half.
        """
        if not text or len(text) <= max_length:
            return text.strip()
        
        # Try to break at sentence end
        sentence_end = text.rfind('.', 0, max_length)
        if sentence_end > max_length // 2:  # Only if we find a sentence break in the latter half
            return text[:sentence_end + 1].strip()
        
        # Try to break at word boundary
        word_boundary = text.rfind(' ', 0, max_length)
        if word_boundary > max_length // 2:  # Only if we find a word break in the latter half
            return text[:word_boundary].strip() + "..."
        
        # Last resort: hard truncate
        return text[:max_length].strip() + "..."
    
    
    def _load_exploitation_patterns(self) -> Dict[ExploitationType, List[Dict]]:
        """Load patterns that indicate exploitation"""
        return {
            ExploitationType.FINANCIAL_EXPLOITATION: [
                {
                    "pattern": r"(?i)security\s+deposit.*?(\d+)\s*times.*?rent|deposit.*?₹?\s*(\d{1,2}),?(\d{2,3}),?(\d{3})",
                    "threshold": 300000,  # More than 3 lakh is suspicious
                    "severity": 80,
                    "description": "Excessive security deposit (should be 2-3 months rent max)"
                },
                {
                    "pattern": r"(?i)rent.*?increase.*?(\d+)%|(\d+)%.*?increase",
                    "threshold": 15,  # More than 15% increase is unfair
                    "severity": 70,
                    "description": "Excessive rent increase percentage"
                },
                {
                    "pattern": r"(?i)penalty.*?₹?\s*(\d+),?(\d+)|fine.*?₹?\s*(\d+),?(\d+)|(\d+)\s*per\s*day",
                    "threshold": 1000,  # More than ₹1000 daily penalty is excessive
                    "severity": 85,
                    "description": "Excessive daily penalties"
                },
                {
                    "pattern": r"(?i)cash\s+only|no\s+receipts?|receipt.*?not.*?provided",
                    "threshold": 1,
                    "severity": 90,
                    "description": "Cash-only payments without receipts (tax evasion)"
                },
                {
                    "pattern": r"(?i)non-?refundable.*?fee|processing.*?fee.*?₹?\s*(\d+)",
                    "threshold": 10000,
                    "severity": 60,
                    "description": "Non-refundable processing fees"
                }
            ],
            ExploitationType.LEGAL_RIGHTS_VIOLATION: [
                {
                    "pattern": r"(?i)waives?\s+all\s+rights|no\s+rights?|cannot\s+approach\s+court",
                    "threshold": 1,
                    "severity": 95,
                    "description": "Illegal waiver of tenant's legal rights"
                },
                {
                    "pattern": r"(?i)disputes?.*?landlord.*?favor|landlord.*?always.*?right",
                    "threshold": 1,
                    "severity": 90,
                    "description": "Biased dispute resolution favoring landlord"
                }
            ],
            ExploitationType.UNFAIR_TERMINATION: [
                {
                    "pattern": r"(?i)tenant.*?(\d+)\s*months?\s*notice|(\d+)\s*months?.*?notice.*?tenant",
                    "threshold": 3,  # More than 3 months notice is unfair
                    "severity": 70,
                    "description": "Excessive notice period required from tenant"
                },
                {
                    "pattern": r"(?i)landlord.*?(\d+)\s*(?:hours?|days?)\s*notice|(\d+)\s*(?:hours?|days?).*?notice.*?landlord",
                    "threshold": 30,  # Less than 30 days is unfair
                    "severity": 80,
                    "description": "Insufficient notice period from landlord"
                },
                {
                    "pattern": r"(?i)no\s+refund.*?circumstances|deposit.*?forfeited",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Unfair deposit forfeiture clause"
                }
            ],
            ExploitationType.PRIVACY_VIOLATION: [
                {
                    "pattern": r"(?i)enter.*?anytime|inspection.*?without.*?notice|landlord.*?enter.*?without",
                    "threshold": 1,
                    "severity": 75,
                    "description": "Landlord can enter without proper notice"
                },
                {
                    "pattern": r"(?i)no\s+guests?.*?after.*?(\d+)|guests?.*?not.*?allowed.*?(\d+)",
                    "threshold": 22,  # Guest restrictions after 10 PM might be reasonable
                    "severity": 60,
                    "description": "Unreasonable guest restrictions"
                }
            ],
            ExploitationType.MAINTENANCE_BURDEN: [
                {
                    "pattern": r"(?i)tenant.*?responsible.*?all.*?repairs|tenant.*?pay.*?structural",
                    "threshold": 1,
                    "severity": 80,
                    "description": "Tenant responsible for structural repairs (landlord's duty)"
                },
                {
                    "pattern": r"(?i)tenant.*?property\s+tax|tenant.*?society\s+charges",
                    "threshold": 1,
                    "severity": 70,
                    "description": "Tenant paying property taxes/society charges"
                }
            ],
            ExploitationType.DISCRIMINATORY_TERMS: [
                {
                    "pattern": r"(?i)no.*?non-?vegetarian|vegetarian.*?only|no.*?meat",
                    "threshold": 1,
                    "severity": 40,
                    "description": "Dietary restrictions (potentially discriminatory)"
                },
                {
                    "pattern": r"(?i)no.*?pets|pets.*?not.*?allowed",
                    "threshold": 1,
                    "severity": 20,
                    "description": "Complete pet ban (even small pets)"
                }
            ],
            # GENERAL CONTRACT EXPLOITATION PATTERNS (work across all contract types)
            ExploitationType.POWER_IMBALANCE: [
                {
                    "pattern": r"(?i)waives?\s+(?:all\s+)?rights?|waives?\s+right\s+to|cannot\s+(?:challenge|dispute|appeal)|no\s+right\s+to|forfeits?\s+(?:all\s+)?rights?",
                    "threshold": 1,
                    "severity": 95,
                    "description": "Illegal waiver of fundamental legal rights"
                },
                {
                    "pattern": r"(?i)(?:company|employer|landlord|party).*?(?:sole\s+discretion|absolute\s+discretion|at\s+will|as\s+(?:they\s+)?see\s+fit)",
                    "threshold": 1,
                    "severity": 80,
                    "description": "One party has absolute discretionary power"
                },
                {
                    "pattern": r"(?i)cannot\s+(?:negotiate|request\s+changes|modify|discuss)|non-?negotiable|take\s+it\s+or\s+leave\s+it",
                    "threshold": 1,
                    "severity": 75,
                    "description": "Contract terms are non-negotiable"
                }
            ],
            ExploitationType.UNCONSCIONABLE_TERMS: [
                {
                    "pattern": r"(?i)penalty.*?(\d+)%|(\d+)%.*?(?:penalty|fine|deduction)|(?:penalty|fine).*?₹?\s*(\d{1,3}),?(\d{3})",
                    "threshold": 25,  # More than 25% penalty is excessive
                    "severity": 85,
                    "description": "Excessive penalty percentages or amounts"
                },
                {
                    "pattern": r"(?i)unlimited.*?(?:liability|obligation|responsibility)|(?:personally\s+)?guarantees?\s+(?:all\s+)?(?:debts?|obligations?|liabilities?)",
                    "threshold": 1,
                    "severity": 90,
                    "description": "Unlimited personal liability or guarantees"
                },
                {
                    "pattern": r"(?i)(?:may\s+be\s+)?(?:delayed|withheld|suspended).*?(?:up\s+to\s+)?(\d+)\s*(?:days?|months?)|payments?.*?delayed.*?(\d+)",
                    "threshold": 30,  # More than 30 days delay is problematic
                    "severity": 70,
                    "description": "Excessive payment delays"
                }
            ],
            ExploitationType.MODIFICATION_ABUSE: [
                {
                    "pattern": r"(?i)(?:may\s+)?(?:modify|change|alter|update).*?(?:any\s+time|without\s+(?:notice|consent))|(?:terms?|conditions?).*?(?:may\s+)?(?:be\s+)?(?:changed|modified).*?(?:any\s+time|unilaterally)",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Contract can be modified unilaterally without consent"
                },
                {
                    "pattern": r"(?i)continued.*?(?:employment|use|occupancy|participation).*?constitutes\s+acceptance|acceptance\s+(?:of\s+)?(?:all\s+)?(?:changes|modifications)",
                    "threshold": 1,
                    "severity": 80,
                    "description": "Forced acceptance of changes through continued relationship"
                }
            ],
            ExploitationType.DISPUTE_RESOLUTION_BIAS: [
                {
                    "pattern": r"(?i)(?:mandatory\s+)?arbitration.*?(?:at\s+)?(?:employee|tenant|party).*?expense|(?:waives?\s+)?right\s+to.*?(?:jury\s+trial|court|litigation)",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Biased dispute resolution requiring party to pay costs"
                },
                {
                    "pattern": r"(?i)(?:jurisdiction|venue).*?chosen.*?(?:solely\s+)?by|(?:all\s+)?(?:legal\s+)?costs?.*?(?:regardless\s+of\s+outcome|borne\s+by)",
                    "threshold": 1,
                    "severity": 75,
                    "description": "Unfair jurisdiction choice or cost allocation"
                }
            ],
            ExploitationType.INTELLECTUAL_PROPERTY_OVERREACH: [
                {
                    "pattern": r"(?i)(?:all\s+)?(?:ideas?|thoughts?|concepts?|inventions?).*?(?:belong\s+to|property\s+of)|(?:personal\s+)?(?:projects?|work).*?(?:outside\s+)?(?:office\s+)?hours?.*?(?:belong|property)",
                    "threshold": 1,
                    "severity": 80,
                    "description": "Overreach of intellectual property rights to personal work"
                },
                {
                    "pattern": r"(?i)(?:existing|prior).*?intellectual\s+property.*?(?:assign|transfer)|(?:future\s+)?(?:patents?|trademarks?|copyrights?).*?(?:automatically\s+)?transfer",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Forced assignment of existing or future intellectual property"
                }
            ],
            ExploitationType.NON_COMPETE_ABUSE: [
                {
                    "pattern": r"(?i)(?:non-?compete|cannot\s+work).*?(\d+)\s*years?|(\d+)\s*years?.*?(?:non-?compete|cannot\s+work)",
                    "threshold": 24,  # More than 2 years is excessive
                    "severity": 85,
                    "description": "Excessive non-compete period duration"
                },
                {
                    "pattern": r"(?i)(?:across\s+)?(?:all\s+)?industries?|(?:any\s+)?(?:company|business|industry)|(?:cannot\s+)?(?:start|begin|engage).*?(?:any\s+)?(?:business|freelance)",
                    "threshold": 1,
                    "severity": 90,
                    "description": "Overly broad non-compete restrictions"
                }
            ],
            ExploitationType.SALARY_MANIPULATION: [
                {
                    "pattern": r"(?i)(?:salary|wages?|compensation).*?(?:subject\s+to\s+)?(?:unlimited\s+)?deductions?|deduct(?:ed|ions?).*?(?:salary|wages?)",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Unlimited or excessive salary deductions"
                },
                {
                    "pattern": r"(?i)(?:sick\s+)?(?:days?|leave).*?(?:penalty|deduction)|(?:penalty|deduction).*?(?:sick\s+)?(?:days?|leave)",
                    "threshold": 1,
                    "severity": 90,
                    "description": "Penalties for taking sick leave (likely illegal)"
                },
                {
                    "pattern": r"(?i)(?:salary|wages?).*?(?:may\s+be\s+)?reduced.*?(?:discretion|any\s+time)|reduced.*?(?:salary|wages?|compensation)",
                    "threshold": 1,
                    "severity": 80,
                    "description": "Arbitrary salary reduction clauses"
                },
                {
                    "pattern": r"(?i)(?:late|tardy|delay).*?(?:arrival|coming).*?(?:deduction|penalty)|(?:full\s+day|entire\s+day).*?(?:salary|pay).*?deduction",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Excessive penalties for minor tardiness"
                },
                {
                    "pattern": r"(?i)(?:unsatisfactory|poor|inadequate).*?(?:performance|work|project).*?(?:deduction|penalty)|(?:₹?\s*)?(\d{1,2}),?(\d{3,})\s*(?:deduction|penalty)",
                    "threshold": 25000,  # More than ₹25,000 deduction is excessive
                    "severity": 80,
                    "description": "Excessive financial penalties for subjective performance issues"
                },
                {
                    "pattern": r"(?i)(?:salary|wages?).*?(?:delayed|withheld|suspended).*?(\d+)\s*days?|payments?.*?(?:up\s+to\s+)?(\d+)\s*days?.*?(?:delay|late)",
                    "threshold": 7,  # More than 7 days delay is problematic
                    "severity": 75,
                    "description": "Excessive salary payment delays"
                }
            ],
            ExploitationType.WORK_CONDITION_VIOLATION: [
                {
                    "pattern": r"(?i)(?:mandatory|required).*?(\d+)\+?\s*hours?.*?(?:week|weekly)|(\d+)\+?\s*hours?.*?(?:per\s+)?week.*?(?:mandatory|required)",
                    "threshold": 50,  # More than 50 hours per week
                    "severity": 75,
                    "description": "Excessive mandatory work hours"
                },
                {
                    "pattern": r"(?i)(?:unlimited\s+)?overtime.*?(?:no\s+)?(?:additional\s+)?compensation|(?:no\s+)?(?:overtime\s+)?(?:pay|compensation).*?overtime",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Unpaid overtime requirements (likely illegal)"
                },
                {
                    "pattern": r"(?i)(?:available\s+)?24/?7|(?:respond|reply).*?(\d+)\s*minutes?|(\d+)\s*minutes?.*?(?:respond|reply)",
                    "threshold": 30,  # Less than 30 minutes response time
                    "severity": 70,
                    "description": "Unreasonable availability or response time requirements"
                },
                {
                    "pattern": r"(?i)(?:all\s+)?(?:holidays?|weekends?).*?(?:mandatory|required|work)|(?:mandatory|required).*?(?:holidays?|weekends?|work)",
                    "threshold": 1,
                    "severity": 80,
                    "description": "Mandatory work on all holidays and weekends"
                },
                {
                    "pattern": r"(?i)(?:vacation|time\s+off|leave).*?(?:may\s+be\s+)?(?:denied|rejected).*?(?:without\s+reason|any\s+reason)|(?:no\s+)?(?:guaranteed|assured).*?(?:vacation|leave)",
                    "threshold": 1,
                    "severity": 75,
                    "description": "Vacation time can be denied without reason"
                },
                {
                    "pattern": r"(?i)(?:probation|probationary).*?(\d+)\s*(?:months?|years?)|(\d+)\s*(?:months?|years?).*?(?:probation|probationary)",
                    "threshold": 12,  # More than 12 months probation
                    "severity": 65,
                    "description": "Excessive probationary period"
                },
                {
                    "pattern": r"(?i)(?:employee|worker).*?(?:purchase|buy).*?(?:stock|shares)|(?:must\s+)?(?:purchase|buy).*?(?:company\s+)?(?:stock|shares)",
                    "threshold": 1,
                    "severity": 85,
                    "description": "Forced purchase of company stock"
                }
            ]
        }
    
    def _load_fair_clause_patterns(self) -> List[Dict]:
        """Load patterns that indicate fair clauses"""
        return [
            {
                "pattern": r"(?i)reasonable\s+notice|30\s+days?\s+notice|proper\s+notice",
                "points": 10,
                "description": "Reasonable notice periods"
            },
            {
                "pattern": r"(?i)security.*?deposit.*?interest|interest.*?deposit",
                "points": 15,
                "description": "Security deposit earns interest"
            },
            {
                "pattern": r"(?i)tenant.*?rights?|rent.*?control.*?act|legal.*?protections?",
                "points": 20,
                "description": "Acknowledges tenant rights"
            },
            {
                "pattern": r"(?i)receipt.*?provided|proper.*?documentation|written.*?receipt",
                "points": 15,
                "description": "Proper documentation and receipts"
            },
            {
                "pattern": r"(?i)normal.*?wear.*?tear|reasonable.*?use",
                "points": 10,
                "description": "Acknowledges normal wear and tear"
            }
        ]
    
    def _load_required_protections(self) -> List[str]:
        """Load list of protections that should be present"""
        return [
            "Proper notice period for termination (30+ days)",
            "Security deposit should not exceed 3 months rent",
            "Landlord responsible for structural repairs",
            "Tenant privacy rights respected",
            "Clear dispute resolution process",
            "Receipt for all payments",
            "Interest on security deposit",
            "Protection under Rent Control Act",
            "Reasonable rent increase limits",
            "Fair maintenance cost sharing"
        ]
    
    async def analyze_contract(self, contract_text: str, ideal_template_text: str = None) -> GuardianScoreResult:
        """
        Analyze a contract and return Guardian Score with detailed exploitation flags
        """
        exploitation_flags = []
        fair_clauses = []
        
        # Detect exploitation patterns
        for exploitation_type, patterns in self.exploitation_patterns.items():
            for pattern_config in patterns:
                matches = re.finditer(pattern_config["pattern"], contract_text)
                for match in matches:
                    # Extract numeric values for threshold checking
                    numeric_values = [int(g) for g in match.groups() if g and g.isdigit()]
                    
                    # Check if values exceed threshold
                    threshold_exceeded = False
                    if numeric_values:
                        if exploitation_type == ExploitationType.UNFAIR_TERMINATION and "tenant" in match.group().lower():
                            # For tenant notice period, higher is worse
                            threshold_exceeded = any(val > pattern_config["threshold"] for val in numeric_values)
                        elif exploitation_type == ExploitationType.UNFAIR_TERMINATION and "landlord" in match.group().lower():
                            # For landlord notice period, lower is worse
                            threshold_exceeded = any(val < pattern_config["threshold"] for val in numeric_values)
                        elif exploitation_type == ExploitationType.PRIVACY_VIOLATION and "after" in match.group().lower():
                            # For guest restrictions, earlier times are worse
                            threshold_exceeded = any(val < pattern_config["threshold"] for val in numeric_values)
                        else:
                            # For most cases, higher values are worse
                            threshold_exceeded = any(val > pattern_config["threshold"] for val in numeric_values)
                    else:
                        # Pattern matched but no numbers (e.g., "waives all rights")
                        threshold_exceeded = True
                    
                    if threshold_exceeded:
                        risk_level = self._determine_risk_level(pattern_config["severity"])
                        
                        flag = ExploitationFlag(
                            type=exploitation_type,
                            risk_level=risk_level,
                            description=pattern_config["description"],
                            clause_text=self._safe_truncate_clause_text(match.group()),
                            severity_score=pattern_config["severity"],
                            recommendation=self._get_recommendation(exploitation_type, pattern_config["description"])
                        )
                        exploitation_flags.append(flag)
        
        # Detect fair clauses
        for fair_pattern in self.fair_clause_patterns:
            if re.search(fair_pattern["pattern"], contract_text, re.IGNORECASE):
                fair_clauses.append(fair_pattern["description"])
        
        # Check for missing protections
        missing_protections = []
        for protection in self.required_protections:
            if not self._has_protection(contract_text, protection):
                missing_protections.append(protection)
        
        # LLM-based harmful clause detection
        llm_flags = await self._llm_based_harmful_clause_detection(contract_text)
        exploitation_flags.extend(llm_flags)
        
        # Use static recommendations for all flags
        for flag in exploitation_flags:
            flag.ai_recommendation = flag.recommendation
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(exploitation_flags, fair_clauses, missing_protections)
        
        # Determine overall risk level
        overall_risk = self._determine_overall_risk(overall_score)
        
        # Generate summary
        summary = self._generate_summary(overall_score, len(exploitation_flags), len(fair_clauses), len(missing_protections))
        
        return GuardianScoreResult(
            overall_score=overall_score,
            risk_level=overall_risk,
            exploitation_flags=exploitation_flags,
            missing_protections=missing_protections,
            fair_clauses=fair_clauses,
            summary=summary
        )
    
    def _determine_risk_level(self, severity: int) -> RiskLevel:
        """Determine risk level based on severity score"""
        if severity >= 85:
            return RiskLevel.CRITICAL
        elif severity >= 70:
            return RiskLevel.HIGH
        elif severity >= 50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_recommendation(self, exploitation_type: ExploitationType, description: str) -> str:
        """Get recommendation for fixing the exploitation"""
        recommendations = {
            ExploitationType.FINANCIAL_EXPLOITATION: "Negotiate reasonable financial terms. Security deposit should not exceed 2-3 months rent.",
            ExploitationType.LEGAL_RIGHTS_VIOLATION: "DO NOT SIGN. This clause violates your legal rights under Indian law.",
            ExploitationType.UNFAIR_TERMINATION: "Negotiate balanced termination clauses with equal notice periods for both parties.",
            ExploitationType.EXCESSIVE_PENALTIES: "Remove excessive penalty clauses. Penalties should be reasonable and proportionate.",
            ExploitationType.MAINTENANCE_BURDEN: "Landlord should be responsible for structural repairs and major maintenance.",
            ExploitationType.PRIVACY_VIOLATION: "Ensure landlord provides 24-48 hours notice before entering the property.",
            ExploitationType.DISCRIMINATORY_TERMS: "Consider if these restrictions are reasonable and legally enforceable.",
            # New general recommendations
            ExploitationType.POWER_IMBALANCE: "DO NOT SIGN. Seek legal counsel immediately. This creates dangerous power imbalance.",
            ExploitationType.UNCONSCIONABLE_TERMS: "REJECT these terms. They are potentially legally unconscionable and unenforceable.",
            ExploitationType.MODIFICATION_ABUSE: "Require written consent for any changes. Unilateral modification clauses are problematic.",
            ExploitationType.DISPUTE_RESOLUTION_BIAS: "Negotiate fair dispute resolution. Both parties should share arbitration costs.",
            ExploitationType.INTELLECTUAL_PROPERTY_OVERREACH: "Limit IP assignment to work-related inventions only. Protect personal projects.",
            ExploitationType.NON_COMPETE_ABUSE: "Reduce non-compete scope and duration. Overly broad restrictions may be unenforceable.",
            ExploitationType.SALARY_MANIPULATION: "CRITICAL: Ensure salary protection. Unlimited deductions violate labor laws.",
            ExploitationType.WORK_CONDITION_VIOLATION: "ILLEGAL: These work conditions likely violate labor laws. Seek legal advice."
        }
        return recommendations.get(exploitation_type, "Review this clause carefully and consider negotiating.")
    
    def _has_protection(self, contract_text: str, protection: str) -> bool:
        """Check if contract has a specific protection"""
        protection_patterns = {
            "Proper notice period": r"(?i)30.*?days?.*?notice|reasonable.*?notice",
            "Security deposit": r"(?i)security.*?deposit.*?(?:2|3|two|three).*?months?",
            "Landlord responsible": r"(?i)landlord.*?responsible.*?(?:structural|major|repairs)",
            "Tenant privacy": r"(?i)(?:24|48).*?hours?.*?notice.*?enter|privacy.*?rights?",
            "dispute resolution": r"(?i)dispute.*?resolution|arbitration|legal.*?process",
            "Receipt": r"(?i)receipt.*?provided|written.*?receipt|proper.*?documentation",
            "Interest": r"(?i)interest.*?deposit|deposit.*?interest",
            "Rent Control": r"(?i)rent.*?control.*?act|tenant.*?rights?",
            "rent increase": r"(?i)rent.*?increase.*?(?:10|annual|yearly)|reasonable.*?increase",
            "maintenance": r"(?i)landlord.*?maintenance|shared.*?maintenance"
        }
        
        for pattern_key, pattern in protection_patterns.items():
            if pattern_key.lower() in protection.lower():
                if re.search(pattern, contract_text):
                    return True
        return False
    
    def _calculate_overall_score(self, exploitation_flags: List[ExploitationFlag], 
                               fair_clauses: List[str], missing_protections: List[str]) -> int:
        """Calculate overall Guardian Score (0-100)"""
        base_score = 100
        
        # Deduct points for exploitation flags
        for flag in exploitation_flags:
            deduction = flag.severity_score * 0.8  # Convert severity to score deduction
            base_score -= deduction
        
        # Deduct points for missing protections
        protection_penalty = len(missing_protections) * 5
        base_score -= protection_penalty
        
        # Add points for fair clauses
        fair_bonus = len(fair_clauses) * 3
        base_score += fair_bonus
        
        # Ensure score is within bounds (minimum 10)
        return max(10, min(100, int(base_score)))
    
    def _determine_overall_risk(self, score: int) -> RiskLevel:
        """Determine overall risk level based on Guardian Score"""
        if score <= 20:
            return RiskLevel.CRITICAL
        elif score <= 40:
            return RiskLevel.HIGH
        elif score <= 60:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_summary(self, score: int, num_flags: int, num_fair: int, num_missing: int) -> str:
        """Generate a human-readable summary"""
        if score <= 20:
            return f"DANGEROUS CONTRACT: This agreement contains {num_flags} serious exploitation issues. DO NOT SIGN without major revisions."
        elif score <= 40:
            return f"HIGH RISK: Found {num_flags} concerning clauses that could be exploitative. Negotiate these terms before signing."
        elif score <= 60:
            return f"MODERATE RISK: Some unfair terms detected. Review {num_flags} flagged clauses and negotiate improvements."
        elif score <= 80:
            return f"ACCEPTABLE: Generally fair contract with {num_fair} good clauses, but could be improved."
        else:
            return f"EXCELLENT: Very fair contract with strong tenant protections. {num_fair} positive clauses identified."