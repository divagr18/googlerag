# api/core/simple_ai_model.py
import logging
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

class SimpleAIModel:
    """
    Simple AI model interface for generating contract recommendations.
    This is a basic implementation that could be enhanced with actual LLM integration.
    """
    
    def __init__(self):
        self.model_name = "simple_recommendation_ai"
        logger.info("Simple AI Model initialized for contract recommendations")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate a contextual response based on the prompt.
        This is a template-based approach that could be enhanced with actual LLM.
        """
        try:
            # Extract key information from the prompt
            clause_info = self._extract_clause_info(prompt)
            
            # Generate contextual recommendation
            recommendation = self._generate_contextual_recommendation(clause_info)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            return "This clause requires careful review. Consider consulting a legal expert for specific advice."
    
    def _extract_clause_info(self, prompt: str) -> dict:
        """Extract key information from the prompt"""
        info = {
            "clause_text": "",
            "issue_type": "general",
            "severity": "medium"
        }
        
        # Extract clause text
        if 'CLAUSE: "' in prompt:
            start = prompt.find('CLAUSE: "') + 9
            end = prompt.find('"', start)
            if end > start:
                info["clause_text"] = prompt[start:end]
        
        # Determine issue type
        if "financial" in prompt.lower():
            info["issue_type"] = "financial"
        elif "legal rights" in prompt.lower() or "illegal" in prompt.lower():
            info["issue_type"] = "legal"
        elif "termination" in prompt.lower():
            info["issue_type"] = "termination"
        elif "privacy" in prompt.lower():
            info["issue_type"] = "privacy"
        elif "maintenance" in prompt.lower():
            info["issue_type"] = "maintenance"
        elif "discriminatory" in prompt.lower():
            info["issue_type"] = "discriminatory"
        
        # Determine severity
        if "CRITICAL" in prompt or "illegal" in prompt.lower():
            info["severity"] = "critical"
        elif "HIGH" in prompt:
            info["severity"] = "high"
        elif "MEDIUM" in prompt:
            info["severity"] = "medium"
        
        return info
    
    def _generate_contextual_recommendation(self, clause_info: dict) -> str:
        """Generate contextual recommendation based on clause information"""
        
        templates = {
            "financial": {
                "critical": "🚨 URGENT: This financial clause is extremely problematic. {issue_specific} This could lead to significant financial loss. Demand removal of this clause or walk away from this agreement. Fair alternatives: {alternatives}",
                "high": "⚠️ WARNING: This financial term is unfair and exploitative. {issue_specific} Negotiate this clause before signing. Suggested changes: {alternatives}",
                "medium": "💰 CAUTION: This financial clause needs improvement. {issue_specific} Consider negotiating: {alternatives}"
            },
            "legal": {
                "critical": "🚫 DO NOT SIGN: This clause violates your fundamental legal rights under Indian law. {issue_specific} Such clauses are often unenforceable and potentially illegal. Seek legal consultation immediately.",
                "high": "⚖️ LEGAL CONCERN: This clause may violate your rights. {issue_specific} Consider consulting a lawyer before agreeing to these terms.",
                "medium": "📋 REVIEW NEEDED: This clause affects your legal standing. {issue_specific} Understand your rights before proceeding."
            },
            "termination": {
                "critical": "🏃‍♂️ UNFAIR TERMINATION: This clause creates extremely one-sided termination terms. {issue_specific} Demand balanced notice periods and fair deposit return policies.",
                "high": "⚖️ IMBALANCED: Termination terms favor the landlord heavily. {issue_specific} Negotiate equal notice periods (30+ days for both parties).",
                "medium": "📅 NEGOTIATE: Termination terms could be more balanced. {issue_specific} Suggest mutual notice periods and clear procedures."
            },
            "privacy": {
                "critical": "🔒 PRIVACY VIOLATION: This clause eliminates your right to privacy. {issue_specific} Landlords must provide 24-48 hours notice except for emergencies.",
                "high": "🏠 PRIVACY CONCERN: Your right to peaceful enjoyment is compromised. {issue_specific} Negotiate proper notice requirements.",
                "medium": "👀 REVIEW ACCESS: Entry terms should be clearer. {issue_specific} Establish reasonable notice periods."
            },
            "maintenance": {
                "critical": "🔧 UNFAIR BURDEN: You're being made responsible for landlord duties. {issue_specific} Structural repairs and major maintenance are landlord responsibilities.",
                "high": "🏗️ MAINTENANCE ISSUE: This unfairly shifts maintenance costs to you. {issue_specific} Negotiate fair cost-sharing arrangements.",
                "medium": "🔨 CLARIFY DUTIES: Maintenance responsibilities need clearer definition. {issue_specific} Establish who pays for what."
            },
            "discriminatory": {
                "critical": "❌ POTENTIALLY ILLEGAL: This restriction may be discriminatory. {issue_specific} Some clauses may violate anti-discrimination laws.",
                "high": "🚩 CONCERNING RESTRICTION: This limitation seems unreasonable. {issue_specific} Consider if this is enforceable or necessary.",
                "medium": "📝 REVIEW RESTRICTION: This limitation should be evaluated. {issue_specific} Understand the reasoning and legality."
            }
        }
        
        # Get appropriate template
        issue_type = clause_info["issue_type"]
        severity = clause_info["severity"]
        
        if issue_type in templates and severity in templates[issue_type]:
            template = templates[issue_type][severity]
        else:
            template = "⚠️ This clause requires careful review. {issue_specific} Consider the implications and negotiate if needed."
        
        # Generate issue-specific content
        issue_specific = self._get_issue_specific_advice(clause_info)
        alternatives = self._get_alternative_suggestions(clause_info)
        
        # Format the recommendation
        recommendation = template.format(
            issue_specific=issue_specific,
            alternatives=alternatives
        )
        
        return recommendation
    
    def _get_issue_specific_advice(self, clause_info: dict) -> str:
        """Get specific advice based on the clause content"""
        clause_text = clause_info["clause_text"].lower()
        
        if "deposit" in clause_text and ("10" in clause_text or "5" in clause_text):
            return "Security deposits exceeding 3 months rent are excessive and potentially illegal."
        elif "cash only" in clause_text:
            return "Cash-only payments without receipts enable tax evasion and leave you without proof of payment."
        elif "no rights" in clause_text or "waive" in clause_text:
            return "Waiving legal rights is often illegal and definitely not advisable."
        elif "24 hours" in clause_text and "notice" in clause_text:
            return "24-hour notice for termination is insufficient and may be illegal in many jurisdictions."
        elif "anytime" in clause_text and "enter" in clause_text:
            return "Landlords cannot enter without proper notice except in emergencies."
        elif "all repairs" in clause_text:
            return "Tenants are not responsible for structural repairs or major maintenance issues."
        elif "%" in clause_text and "increase" in clause_text:
            return "Rent increases above 10-15% annually are excessive and may violate rent control laws."
        else:
            return "This clause creates an unfair advantage for the landlord."
    
    def _get_alternative_suggestions(self, clause_info: dict) -> str:
        """Suggest fair alternatives"""
        issue_type = clause_info["issue_type"]
        
        alternatives = {
            "financial": "2-3 months security deposit, annual rent increases capped at 10%, all payments with receipts",
            "legal": "Maintain all legal rights, fair dispute resolution through proper legal channels",
            "termination": "30+ days notice for both parties, deposit return within 30 days minus reasonable deductions",
            "privacy": "24-48 hours notice for entry except emergencies, reasonable guest policies",
            "maintenance": "Landlord handles structural/major repairs, tenant handles minor maintenance, shared utility costs",
            "discriminatory": "Reasonable, non-discriminatory restrictions that comply with fair housing laws"
        }
        
        return alternatives.get(issue_type, "Fair, balanced terms that protect both parties' interests")


# Initialize and add to ml_models during startup
def initialize_ai_model():
    """Initialize the AI model and add it to ml_models"""
    from api.state import ml_models
    
    ai_model = SimpleAIModel()
    ml_models["ai_model"] = ai_model
    logger.info("AI model initialized and added to ml_models")
    
    return ai_model