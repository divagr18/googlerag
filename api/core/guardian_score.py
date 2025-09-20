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
    Analyzes rental contracts against ideal templates to detect exploitation
    and protect users from unfair terms.
    """
    
    def __init__(self):
        self.exploitation_patterns = self._load_exploitation_patterns()
        self.fair_clause_patterns = self._load_fair_clause_patterns()
        self.required_protections = self._load_required_protections()
        self.ai_engine = None  # Will be set when AI is available
    
    def set_ai_engine(self, ai_engine):
        """Set the AI recommendation engine"""
        self.ai_engine = ai_engine
    
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
                            clause_text=match.group()[:200] + "..." if len(match.group()) > 200 else match.group(),
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
        
        # Generate AI recommendations for exploitation flags
        if self.ai_engine and exploitation_flags:
            try:
                # Prepare flags for AI recommendation generation
                flag_data = []
                for flag in exploitation_flags:
                    flag_data.append({
                        "type": flag.type.value,
                        "clause_text": flag.clause_text,
                        "description": flag.description,
                        "severity_score": flag.severity_score
                    })
                
                # Generate AI recommendations
                enhanced_flags_data = await self.ai_engine.generate_bulk_recommendations(flag_data)
                
                # Update exploitation flags with AI recommendations
                for i, flag in enumerate(exploitation_flags):
                    if i < len(enhanced_flags_data):
                        flag.ai_recommendation = enhanced_flags_data[i].get("ai_recommendation", flag.recommendation)
                    else:
                        flag.ai_recommendation = flag.recommendation
                        
            except Exception as e:
                logger.warning(f"AI recommendation generation failed: {str(e)}")
                # Continue with static recommendations
                for flag in exploitation_flags:
                    flag.ai_recommendation = flag.recommendation
        else:
            # Use static recommendations if AI is not available
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
            ExploitationType.DISCRIMINATORY_TERMS: "Consider if these restrictions are reasonable and legally enforceable."
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