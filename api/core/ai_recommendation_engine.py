# api/core/ai_recommendation_engine.py
import asyncio
from typing import Dict, List, Optional
import logging
from api.state import ml_models

logger = logging.getLogger(__name__)

class AIRecommendationEngine:
    """
    Generates AI-powered, contextual recommendations for contract exploitation issues
    using LLM to provide personalized advice based on specific clause content.
    """
    
    def __init__(self):
        self.recommendation_prompts = {
            "financial_exploitation": """
You are a legal advisor helping a tenant understand problematic financial clauses in their rental agreement. 

PROBLEMATIC CLAUSE: "{clause_text}"
ISSUE IDENTIFIED: {description}
SEVERITY: {severity_score}/100

Generate a specific, actionable recommendation that:
1. Explains WHY this clause is problematic
2. Provides SPECIFIC negotiation points
3. Suggests ALTERNATIVE fair terms
4. Warns about potential consequences if not changed

Keep the response under 150 words and make it practical for an average person to understand and act upon.
            """,
            
            "legal_rights_violation": """
You are a legal rights advocate helping someone identify illegal clauses in their contract.

ILLEGAL CLAUSE: "{clause_text}"
VIOLATION: {description}
SEVERITY: {severity_score}/100 (CRITICAL)

Generate an urgent, clear recommendation that:
1. States this is ILLEGAL under Indian law
2. Explains the specific legal rights being violated
3. Advises NOT to sign unless removed
4. Mentions relevant laws/protections (Rent Control Act, Consumer Protection Act, etc.)
5. Suggests seeking legal consultation if needed

Be firm and protective - this person's legal rights are at stake. Keep under 150 words.
            """,
            
            "unfair_termination": """
You are helping someone understand unfair termination clauses in their rental agreement.

UNFAIR CLAUSE: "{clause_text}"
PROBLEM: {description}
SEVERITY: {severity_score}/100

Generate a balanced recommendation that:
1. Explains how this clause is one-sided and unfair
2. Suggests fair alternatives (equal notice periods, reasonable deposit terms)
3. Provides negotiation strategies
4. Explains industry standards for comparison

Make it actionable and help them negotiate better terms. Keep under 150 words.
            """,
            
            "excessive_penalties": """
You are advising someone about excessive penalty clauses in their contract.

PENALTY CLAUSE: "{clause_text}"
ISSUE: {description}
SEVERITY: {severity_score}/100

Generate practical advice that:
1. Explains why these penalties are excessive/unreasonable
2. Suggests reasonable penalty amounts
3. Provides negotiation tactics to reduce or remove them
4. Explains what normal wear and tear should cover

Help them negotiate fair penalty terms. Keep under 150 words.
            """,
            
            "maintenance_burden": """
You are helping someone understand unfair maintenance responsibilities in their rental agreement.

MAINTENANCE CLAUSE: "{clause_text}"
PROBLEM: {description}
SEVERITY: {severity_score}/100

Generate clear advice that:
1. Explains which repairs are landlord vs tenant responsibility
2. Clarifies what's normal vs structural maintenance
3. Suggests how to negotiate fair maintenance splitting
4. References standard practices in rental agreements

Help them understand their rights and negotiate fair terms. Keep under 150 words.
            """,
            
            "privacy_violation": """
You are helping someone protect their privacy rights as a tenant.

PRIVACY ISSUE: "{clause_text}"
VIOLATION: {description}
SEVERITY: {severity_score}/100

Generate protective advice that:
1. Explains their right to privacy and peaceful enjoyment
2. Suggests reasonable notice periods (24-48 hours)
3. Provides language for negotiating privacy protections
4. Explains what constitutes reasonable vs unreasonable access

Help them establish proper boundaries with their landlord. Keep under 150 words.
            """,
            
            "discriminatory_terms": """
You are helping someone evaluate potentially discriminatory clauses in their rental agreement.

RESTRICTIVE CLAUSE: "{clause_text}"
CONCERN: {description}
SEVERITY: {severity_score}/100

Generate balanced advice that:
1. Explains whether this restriction is reasonable or potentially discriminatory
2. Discusses legal enforceability of such clauses
3. Suggests alternatives or modifications if problematic
4. Advises on when to seek legal opinion

Help them understand their rights while being practical about housing realities. Keep under 150 words.
            """
        }
    
    async def generate_recommendation(
        self, 
        exploitation_type: str, 
        clause_text: str, 
        description: str, 
        severity_score: int,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate AI-powered recommendation for a specific exploitation issue
        """
        try:
            # Get the appropriate prompt template
            prompt_template = self.recommendation_prompts.get(
                exploitation_type, 
                self.recommendation_prompts["financial_exploitation"]  # fallback
            )
            
            # Format the prompt with specific details
            formatted_prompt = prompt_template.format(
                clause_text=clause_text[:200],  # Limit clause text to 200 chars
                description=description,
                severity_score=severity_score
            )
            
            # Get the AI model from ml_models
            ai_model = ml_models.get("ai_model")
            if not ai_model:
                logger.warning("AI model not available, falling back to static recommendation")
                return self._get_fallback_recommendation(exploitation_type, description)
            
            # Generate AI recommendation
            try:
                response = await ai_model.generate_response(formatted_prompt)
                
                # Clean up the response
                recommendation = response.strip()
                if len(recommendation) > 200:
                    recommendation = recommendation[:200] + "..."
                
                return recommendation
                
            except Exception as e:
                logger.error(f"AI generation failed: {str(e)}")
                return self._get_fallback_recommendation(exploitation_type, description)
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return self._get_fallback_recommendation(exploitation_type, description)
    
    def _get_fallback_recommendation(self, exploitation_type: str, description: str) -> str:
        """
        Fallback recommendations when AI is not available
        """
        fallback_recommendations = {
            "financial_exploitation": f"⚠️ {description} - Negotiate reasonable financial terms. Security deposits should not exceed 2-3 months rent. Request transparent pricing and proper receipts.",
            
            "legal_rights_violation": f"🚫 CRITICAL: {description} - DO NOT SIGN. This clause violates your legal rights under Indian law. Consult a lawyer if needed.",
            
            "unfair_termination": f"⚖️ {description} - Negotiate balanced termination clauses with equal notice periods for both parties (30+ days). Ensure fair deposit return terms.",
            
            "excessive_penalties": f"💰 {description} - Remove excessive penalty clauses. Penalties should be reasonable and proportionate to actual damages caused.",
            
            "maintenance_burden": f"🔧 {description} - Landlord should be responsible for structural repairs and major maintenance. Negotiate fair cost sharing for utilities.",
            
            "privacy_violation": f"🏠 {description} - Ensure landlord provides 24-48 hours notice before entering. You have right to peaceful enjoyment of the property.",
            
            "discriminatory_terms": f"👥 {description} - Consider if these restrictions are reasonable and legally enforceable. Some may be discriminatory or overly restrictive."
        }
        
        return fallback_recommendations.get(
            exploitation_type, 
            f"⚠️ {description} - Review this clause carefully and consider negotiating more favorable terms."
        )
    
    async def generate_bulk_recommendations(
        self, 
        exploitation_flags: List[Dict]
    ) -> List[Dict]:
        """
        Generate recommendations for multiple exploitation flags efficiently
        """
        # Process recommendations in parallel for better performance
        tasks = [
            self.generate_recommendation(
                flag["type"],
                flag["clause_text"],
                flag["description"], 
                flag["severity_score"]
            )
            for flag in exploitation_flags
        ]
        
        recommendations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update flags with AI-generated recommendations
        enhanced_flags = []
        for i, flag in enumerate(exploitation_flags):
            enhanced_flag = flag.copy()
            if isinstance(recommendations[i], str):
                enhanced_flag["ai_recommendation"] = recommendations[i]
            else:
                # If AI generation failed, use fallback
                enhanced_flag["ai_recommendation"] = self._get_fallback_recommendation(
                    flag["type"], flag["description"]
                )
            enhanced_flags.append(enhanced_flag)
        
        return enhanced_flags