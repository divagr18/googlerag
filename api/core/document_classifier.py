# api/core/document_classifier.py
import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Automatically classifies uploaded documents to determine if they are contracts
    and what type of contracts they are.
    """
    
    def __init__(self):
        self.contract_patterns = self._load_contract_patterns()
        self.document_type_patterns = self._load_document_type_patterns()
    
    def _load_contract_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate different contract types"""
        return {
            "rental": [
                r"(?i)rental\s+agreement", r"(?i)lease\s+agreement", r"(?i)tenancy\s+agreement",
                r"(?i)rent\s+(?:deed|contract)", r"(?i)landlord.*tenant", r"(?i)tenant.*landlord",
                r"(?i)premises.*rent", r"(?i)security\s+deposit", r"(?i)monthly\s+rent",
                r"(?i)leave\s+(?:and|&)\s+license", r"(?i)occupation.*premises",
                # Enhanced rental patterns
                r"(?i)\brent\b", r"(?i)\btenant\b", r"(?i)\blandlord\b", r"(?i)\bowner\b",
                r"(?i)\bdeposit\b", r"(?i)\bmaintenance\b", r"(?i)\brepairs\b", r"(?i)\bproperty\b"
            ],
            "employment": [
                r"(?i)employment\s+(?:agreement|contract)", r"(?i)job\s+(?:agreement|contract)",
                r"(?i)work\s+(?:agreement|contract)", r"(?i)service\s+(?:agreement|contract)",
                r"(?i)employer.*employee", r"(?i)employee.*employer", r"(?i)salary.*position",
                r"(?i)terms\s+of\s+employment", r"(?i)job\s+description", r"(?i)probation\s+period",
                # Enhanced employment patterns  
                r"(?i)\bemployee\b", r"(?i)\bemployer\b", r"(?i)\bcompany\b", r"(?i)\bsalary\b",
                r"(?i)\bwages\b", r"(?i)\bpayroll\b", r"(?i)\bworkplace\b", r"(?i)\bovertime\b",
                r"(?i)\btermination\b", r"(?i)\bresignation\b", r"(?i)\bconfidentiality\b",
                r"(?i)non[\-\s]?compete", r"(?i)intellectual\s+property", r"(?i)work\s+hours?",
                # Additional employment-specific patterns
                r"(?i)annual\s+salary", r"(?i)compensation\s+package", r"(?i)benefits\s+package",
                r"(?i)sick\s+leave", r"(?i)vacation\s+days?", r"(?i)paid\s+leave", r"(?i)maternity\s+leave",
                r"(?i)performance\s+review", r"(?i)job\s+responsibilities", r"(?i)reporting\s+(?:to|manager)",
                r"(?i)start\s+date", r"(?i)joining\s+date", r"(?i)full[\-\s]?time", r"(?i)part[\-\s]?time",
                r"(?i)working\s+hours", r"(?i)office\s+hours", r"(?i)remote\s+work", r"(?i)work\s+from\s+home",
                r"(?i)notice\s+period", r"(?i)severance\s+pay", r"(?i)gratuity", r"(?i)pension",
                r"(?i)disciplinary\s+action", r"(?i)code\s+of\s+conduct", r"(?i)dress\s+code",
                r"(?i)training\s+period", r"(?i)induction", r"(?i)orientation",
                r"(?i)stock\s+options", r"(?i)equity", r"(?i)bonus", r"(?i)incentive",
                r"(?i)appraisal", r"(?i)increment", r"(?i)promotion", r"(?i)demotion",
                r"(?i)background\s+check", r"(?i)medical\s+examination", r"(?i)drug\s+test",
                r"(?i)chief\s+(?:marketing|technology|executive|financial)\s+officer",
                r"(?i)manager", r"(?i)director", r"(?i)executive", r"(?i)supervisor",
                r"(?i)team\s+lead", r"(?i)department", r"(?i)division", r"(?i)designation"
            ],
            "nda": [
                r"(?i)non-?disclosure\s+agreement", r"(?i)confidentiality\s+agreement",
                r"(?i)secrecy\s+agreement", r"(?i)confidential\s+information",
                r"(?i)proprietary\s+information", r"(?i)trade\s+secrets",
                r"(?i)disclosure.*prohibited", r"(?i)confidential.*material"
            ],
            "partnership": [
                r"(?i)partnership\s+(?:agreement|deed)", r"(?i)business\s+partnership",
                r"(?i)joint\s+venture", r"(?i)profit\s+sharing", r"(?i)capital\s+contribution",
                r"(?i)partner.*rights", r"(?i)partnership\s+firm"
            ],
            "service": [
                r"(?i)service\s+(?:agreement|contract)", r"(?i)consulting\s+agreement",
                r"(?i)professional\s+services", r"(?i)freelance\s+agreement",
                r"(?i)contractor\s+agreement", r"(?i)scope\s+of\s+work"
            ],
            "sale": [
                r"(?i)sale\s+(?:agreement|deed)", r"(?i)purchase\s+agreement",
                r"(?i)buy.*sell", r"(?i)vendor.*purchaser", r"(?i)sale\s+of\s+goods",
                r"(?i)transfer\s+of\s+ownership", r"(?i)sale\s+consideration",
                # Enhanced purchase/sale patterns
                r"(?i)\bseller\b", r"(?i)\bbuyer\b", r"(?i)\bvendor\b", r"(?i)\bpurchaser\b",
                r"(?i)\bsale\s+price\b", r"(?i)\bpurchase\s+price\b", r"(?i)\bconsideration\b",
                r"(?i)advance\s+payment", r"(?i)down\s+payment", r"(?i)token\s+amount",
                r"(?i)earnest\s+money", r"(?i)booking\s+amount", r"(?i)security\s+deposit",
                r"(?i)possession", r"(?i)delivery", r"(?i)handover", r"(?i)completion\s+date",
                r"(?i)property\s+sale", r"(?i)real\s+estate", r"(?i)immovable\s+property",
                r"(?i)movable\s+property", r"(?i)goods\s+and\s+services", r"(?i)assets",
                r"(?i)title\s+(?:deed|documents?)", r"(?i)ownership\s+documents?", r"(?i)registration",
                r"(?i)stamp\s+duty", r"(?i)registration\s+fee", r"(?i)transfer\s+charges",
                r"(?i)mutation", r"(?i)khata", r"(?i)survey\s+number", r"(?i)plot\s+number",
                r"(?i)apartment", r"(?i)flat", r"(?i)villa", r"(?i)house", r"(?i)land",
                r"(?i)commercial\s+property", r"(?i)residential\s+property", r"(?i)office\s+space",
                r"(?i)shop", r"(?i)warehouse", r"(?i)factory", r"(?i)building",
                r"(?i)square\s+(?:feet|ft)", r"(?i)sq\.?\s*ft", r"(?i)area", r"(?i)carpet\s+area",
                r"(?i)built[\-\s]?up\s+area", r"(?i)super\s+built[\-\s]?up\s+area",
                r"(?i)possession\s+certificate", r"(?i)completion\s+certificate", r"(?i)occupancy\s+certificate",
                r"(?i)approved\s+plan", r"(?i)sanctioned\s+plan", r"(?i)municipal\s+approval",
                r"(?i)clear\s+title", r"(?i)marketable\s+title", r"(?i)encumbrance\s+certificate",
                r"(?i)property\s+tax", r"(?i)maintenance\s+charges", r"(?i)society\s+dues",
                r"(?i)defects\s+liability", r"(?i)warranty", r"(?i)guarantee", r"(?i)defects",
                r"(?i)fixtures\s+and\s+fittings", r"(?i)furnishing", r"(?i)amenities",
                r"(?i)parking\s+space", r"(?i)car\s+park", r"(?i)garage", r"(?i)balcony"
            ],
            "loan": [
                r"(?i)loan\s+agreement", r"(?i)credit\s+agreement", r"(?i)borrower.*lender",
                r"(?i)principal\s+amount", r"(?i)interest\s+rate", r"(?i)repayment\s+schedule",
                r"(?i)mortgage\s+deed", r"(?i)security\s+for\s+loan",
                # Enhanced loan-specific patterns
                r"(?i)\bborrower\b", r"(?i)\blender\b", r"(?i)\bloan\s+amount", r"(?i)\bprincipal\b",
                r"(?i)\binterest\b", r"(?i)\bemi\b", r"(?i)equated\s+monthly\s+installment",
                r"(?i)monthly\s+installment", r"(?i)repayment\s+terms", r"(?i)loan\s+tenure",
                r"(?i)loan\s+period", r"(?i)maturity\s+date", r"(?i)due\s+date",
                r"(?i)default", r"(?i)foreclosure", r"(?i)prepayment", r"(?i)penalty",
                r"(?i)processing\s+fee", r"(?i)administrative\s+charges", r"(?i)late\s+payment",
                r"(?i)collateral", r"(?i)security\s+(?:deposit|guarantee)", r"(?i)guarantor",
                r"(?i)co[\-\s]?borrower", r"(?i)joint\s+borrower", r"(?i)surety",
                r"(?i)personal\s+loan", r"(?i)home\s+loan", r"(?i)business\s+loan", r"(?i)education\s+loan",
                r"(?i)vehicle\s+loan", r"(?i)car\s+loan", r"(?i)bike\s+loan", r"(?i)gold\s+loan",
                r"(?i)mortgage", r"(?i)hypothecation", r"(?i)pledge", r"(?i)lien",
                r"(?i)bank", r"(?i)nbfc", r"(?i)financial\s+institution", r"(?i)credit\s+score",
                r"(?i)cibil", r"(?i)credit\s+history", r"(?i)income\s+proof", r"(?i)salary\s+slip",
                r"(?i)disbursement", r"(?i)sanction", r"(?i)approval", r"(?i)credit\s+limit",
                r"(?i)outstanding\s+amount", r"(?i)balance\s+transfer", r"(?i)top[\-\s]?up",
                r"(?i)floating\s+rate", r"(?i)fixed\s+rate", r"(?i)rate\s+of\s+interest",
                r"(?i)compound\s+interest", r"(?i)simple\s+interest", r"(?i)annual\s+percentage\s+rate",
                r"(?i)apr", r"(?i)rbi", r"(?i)reserve\s+bank", r"(?i)banking\s+regulation"
            ],
            "purchase": [
                # Purchase-specific patterns (buyer's perspective)
                r"(?i)purchase\s+(?:agreement|contract|deed)", r"(?i)buying\s+agreement",
                r"(?i)acquisition\s+agreement", r"(?i)procurement\s+contract",
                r"(?i)\bbuyer\b", r"(?i)\bpurchaser\b", r"(?i)\bacquirer\b",
                r"(?i)purchase\s+order", r"(?i)purchase\s+price", r"(?i)buying\s+price",
                r"(?i)procurement", r"(?i)acquisition", r"(?i)sourcing",
                r"(?i)vendor\s+selection", r"(?i)supplier\s+agreement", r"(?i)goods\s+purchase",
                r"(?i)equipment\s+purchase", r"(?i)bulk\s+purchase", r"(?i)wholesale\s+purchase"
            ],
            "property_sale": [
                # Property-specific sale patterns
                r"(?i)property\s+(?:sale|purchase)", r"(?i)real\s+estate\s+(?:sale|transaction)",
                r"(?i)(?:house|home|apartment|flat|villa)\s+(?:sale|purchase)",
                r"(?i)land\s+(?:sale|purchase)", r"(?i)plot\s+(?:sale|purchase)",
                r"(?i)property\s+transfer", r"(?i)real\s+estate\s+transfer",
                r"(?i)conveyance\s+deed", r"(?i)sale\s+deed", r"(?i)property\s+deed",
                r"(?i)immovable\s+property", r"(?i)real\s+property", r"(?i)realty",
                r"(?i)property\s+registration", r"(?i)sub[\-\s]?registrar", r"(?i)registrar\s+office"
            ],
            "loan_agreement": [
                # Alternative naming for loan contracts
                r"(?i)loan\s+agreement", r"(?i)lending\s+agreement", r"(?i)credit\s+facility",
                r"(?i)financial\s+assistance", r"(?i)borrowing\s+agreement", r"(?i)advance\s+agreement"
            ]
        }
    
    def _load_document_type_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate if document is a contract"""
        return {
            "contract_indicators": [
                # Common contract words (case insensitive)
                r"(?i)\bagreement\b", r"(?i)\bcontract\b", r"(?i)\bdeed\b", r"(?i)\bmemorandum\b",
                r"(?i)\bterms\s+(?:and|&)\s+conditions\b", r"(?i)\bwhereas\b", r"(?i)\bparty.*first\b",
                r"(?i)\bparty.*second\b", r"(?i)\bwitnesseth\b", r"(?i)\bconsideration\b",
                r"(?i)this\s+agreement.*made", r"(?i)entered\s+into", r"(?i)\bbinding\b",
                r"(?i)\bobligations\b", r"(?i)\brepresentations\b", r"(?i)\bwarranties\b",
                # Additional common contract terms
                r"(?i)\bemployer\b", r"(?i)\bemployee\b", r"(?i)\btenant\b", r"(?i)\blandlord\b",
                r"(?i)\bowner\b", r"(?i)\brental\b", r"(?i)\blease\b", r"(?i)\brent\b",
                r"(?i)\bdeposit\b", r"(?i)\bsecurity\s+deposit\b", r"(?i)\btermination\b",
                r"(?i)\bclause\b", r"(?i)\bsection\b", r"(?i)\barticle\b"
            ],
            "legal_language": [
                r"(?i)hereby\s+agree", r"(?i)in\s+witness\s+whereof", r"(?i)force\s+and\s+effect",
                r"(?i)breach\s+of\s+contract", r"(?i)governing\s+law", r"(?i)\bjurisdiction\b",
                r"(?i)dispute\s+resolution", r"(?i)termination\s+clause", r"(?i)\bindemnity\b",
                # Additional legal terms
                r"(?i)\bshall\b", r"(?i)\bmay\s+not\b", r"(?i)\bmust\s+not\b", r"(?i)\bliable\b",
                r"(?i)\bresponsible\s+for\b", r"(?i)\bin\s+case\s+of\b", r"(?i)\bsubject\s+to\b",
                r"(?i)\bnotwithstanding\b", r"(?i)\bprovided\s+that\b"
            ],
            "signature_indicators": [
                r"(?i)signature.*party", r"(?i)signed\s+in\s+presence", r"(?i)witness.*signature",
                r"(?i)executed.*day", r"(?i)seal.*signature", r"(?i)thumb\s+impression",
                # Simple signature indicators
                r"(?i)\bsignature\b", r"(?i)\bsigned\b", r"(?i)\bdate\b.*:.*_+",
                r"(?i)employee\s+signature", r"(?i)tenant\s+signature", r"(?i)owner\s+signature"
            ]
        }
    
    def classify_document(self, text: str, filename: str = "") -> Dict[str, any]:
        """
        Classify a document to determine if it's a contract and what type
        
        Returns:
            {
                "is_contract": bool,
                "contract_type": str or None,
                "confidence": float,
                "detected_patterns": List[str],
                "should_analyze": bool
            }
        """
        try:
            # First, determine if this is likely a contract
            is_contract, contract_confidence = self._is_contract(text, filename)
            
            if not is_contract:
                return {
                    "is_contract": False,
                    "contract_type": None,
                    "confidence": contract_confidence,
                    "detected_patterns": [],
                    "should_analyze": False
                }
            
            # If it's a contract, classify the type
            contract_type, type_confidence, patterns = self._classify_contract_type(text, filename)
            
            # Determine if we should run Guardian Score analysis
            # Now analyzing all major contract types that can contain exploitation
            # Based on ideal template categories: rental, employment, partnership, purchase, lease, loan_agreement, nda
            should_analyze = is_contract and contract_type in [
                "rental", "employment", "service", "partnership", "purchase", 
                "lease", "loan_agreement", "loan", "nda", "property_sale"
            ]
            
            return {
                "is_contract": is_contract,
                "contract_type": contract_type,
                "confidence": max(contract_confidence, type_confidence),
                "detected_patterns": patterns,
                "should_analyze": should_analyze
            }
            
        except Exception as e:
            logger.error(f"Document classification failed: {str(e)}")
            return {
                "is_contract": False,
                "contract_type": None,
                "confidence": 0.0,
                "detected_patterns": [],
                "should_analyze": False
            }
    
    def _is_contract(self, text: str, filename: str) -> tuple[bool, float]:
        """Determine if document is likely a contract"""
        score = 0
        max_score = 0
        
        # Check filename for contract indicators
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ["agreement", "contract", "deed", "lease", "rental"]):
            score += 20
        max_score += 20
        
        # Check for contract language patterns
        for category, patterns in self.document_type_patterns.items():
            category_score = 0
            category_max = len(patterns) * 10
            
            for pattern in patterns:
                if re.search(pattern, text):
                    category_score += 10
            
            # Weight different categories
            if category == "contract_indicators":
                score += category_score
                max_score += category_max
            elif category == "legal_language":
                score += category_score * 0.8
                max_score += category_max * 0.8
            elif category == "signature_indicators":
                score += category_score * 0.5
                max_score += category_max * 0.5
        
        confidence = min(score / max_score, 1.0) if max_score > 0 else 0.0
        is_contract = confidence > 0.2  # Lowered from 0.3 to 0.2 (20% threshold)
        
        # Debug logging
        print(f"🔍 Contract detection debug:")
        print(f"   Total score: {score}/{max_score} = {confidence:.3f}")
        print(f"   Is contract: {is_contract} (threshold: 0.2)")
        
        return is_contract, confidence
    
    def _classify_contract_type(self, text: str, filename: str) -> tuple[Optional[str], float, List[str]]:
        """Classify the specific type of contract"""
        type_scores = {}
        detected_patterns = []
        
        # Check filename for type hints
        filename_lower = filename.lower()
        for contract_type in self.contract_patterns.keys():
            if contract_type in filename_lower:
                type_scores[contract_type] = type_scores.get(contract_type, 0) + 30
        
        # Check text content for contract type patterns
        for contract_type, patterns in self.contract_patterns.items():
            type_score = 0
            type_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    type_score += len(matches) * 10
                    type_patterns.extend([pattern.replace(r'(?i)', '').replace(r'\s+', ' ') for _ in matches])
            
            if type_score > 0:
                type_scores[contract_type] = type_scores.get(contract_type, 0) + type_score
                detected_patterns.extend(type_patterns)
        
        if not type_scores:
            return None, 0.0, []
        
        # Find the contract type with highest score
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        best_score = type_scores[best_type]
        
        # Calculate confidence (normalize to 0-1)
        max_possible_score = len(self.contract_patterns[best_type]) * 10 + 30  # patterns + filename bonus
        confidence = min(best_score / max_possible_score, 1.0)
        
        return best_type, confidence, detected_patterns[:5]  # Return top 5 patterns
    
    def get_analysis_recommendation(self, classification: Dict[str, any]) -> str:
        """Get recommendation on whether to run Guardian Score analysis"""
        if not classification["is_contract"]:
            return "Not identified as a contract - no Guardian Score analysis needed."
        
        if classification["should_analyze"]:
            return f"Contract identified as '{classification['contract_type']}' - Guardian Score analysis recommended."
        
        return f"Contract identified as '{classification['contract_type']}' - Guardian Score analysis not available for this contract type yet."