"""
Ideal Contract Manager for Guardian Score System

This module manages ideal contract templates that serve as benchmarks for 
scoring uploaded contracts. Ideal contracts contain essential clauses,
risk factors, and compliance requirements for different contract types.
"""

import os
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
import numpy as np
from enum import Enum


class ContractCategory(str, Enum):
    """Supported contract categories for ideal templates (India-focused)"""
    RENTAL = "rental"  # Leave & License Agreements
    EMPLOYMENT = "employment" 
    SERVICE_AGREEMENT = "service_agreement"
    NDA = "nda"
    PURCHASE = "purchase"
    LEASE = "lease"  # Commercial lease
    PARTNERSHIP = "partnership"  # Partnership Deed
    LICENSING = "licensing"
    CONSULTING = "consulting"
    LLP = "llp"  # Limited Liability Partnership
    PROPERTY_SALE = "property_sale"  # Property transactions
    LOAN_AGREEMENT = "loan_agreement"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk levels for contract clauses"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IdealContractManager:
    """
    Manages ideal contract templates for the Guardian Score system.
    
    Stores ideal contracts with their essential clauses, risk factors,
    and compliance requirements to enable comparison with uploaded contracts.
    """
    
    def __init__(self, storage_path: str = "./ideal_contracts_db"):
        """
        Initialize the Ideal Contract Manager.
        
        Args:
            storage_path: Path to store the ChromaDB collection
        """
        self.storage_path = storage_path
        self.collection_name = "ideal_contracts"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=storage_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection for ideal contracts
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            print(f"📚 Using existing ideal contracts collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"🆕 Created new ideal contracts collection: {self.collection_name}")
    
    def _generate_template_id(self, category: str, title: str) -> str:
        """Generate a unique ID for an ideal contract template."""
        unique_string = f"{category}:{title}:{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def store_ideal_contract(
        self,
        category: ContractCategory,
        title: str,
        description: str,
        essential_clauses: List[Dict],
        risk_factors: List[Dict],
        compliance_requirements: List[Dict],
        scoring_weights: Dict[str, float],
        embedding: np.ndarray,
        created_by: str = "system"
    ) -> str:
        """
        Store an ideal contract template.
        
        Args:
            category: Contract category (rental, employment, etc.)
            title: Title of the ideal contract template
            description: Description of the template
            essential_clauses: List of required clauses with importance scores
            risk_factors: List of risk factors to check for
            compliance_requirements: List of compliance requirements
            scoring_weights: Weights for different scoring components
            embedding: Text embedding for the template
            created_by: User who created this template
            
        Returns:
            Template ID
            
        Example essential_clauses:
        [
            {
                "name": "rent_amount",
                "description": "Monthly rent amount must be clearly specified", 
                "importance": 10,  # Scale 1-10
                "keywords": ["rent", "monthly payment", "amount"],
                "required": True
            },
            {
                "name": "security_deposit",
                "description": "Security deposit terms and conditions",
                "importance": 8,
                "keywords": ["security deposit", "damage deposit"],
                "required": True
            }
        ]
        
        Example risk_factors:
        [
            {
                "name": "automatic_renewal",
                "description": "Automatic renewal without notice",
                "risk_level": "high",
                "keywords": ["automatic renewal", "auto-renew"],
                "penalty_score": -15
            }
        ]
        """
        template_id = self._generate_template_id(category.value, title)
        
        # Prepare metadata
        metadata = {
            "template_id": template_id,
            "category": category.value,
            "title": title,
            "description": description,
            "created_by": created_by,
            "created_timestamp": datetime.now().isoformat(),
            "processing_version": "v1.0",
            "total_essential_clauses": len(essential_clauses),
            "total_risk_factors": len(risk_factors),
            "total_compliance_requirements": len(compliance_requirements)
        }
        
        # Store the full template data as document text (JSON)
        template_data = {
            "template_id": template_id,
            "category": category.value,
            "title": title,
            "description": description,
            "essential_clauses": essential_clauses,
            "risk_factors": risk_factors,
            "compliance_requirements": compliance_requirements,
            "scoring_weights": scoring_weights,
            "metadata": metadata
        }
        
        document_text = json.dumps(template_data, indent=2)
        
        try:
            # Store in ChromaDB
            self.collection.add(
                ids=[template_id],
                documents=[document_text],
                metadatas=[metadata],
                embeddings=[embedding.tolist()]
            )
            
            print(f"✅ Stored ideal contract template: {title} ({category.value})")
            return template_id
            
        except Exception as e:
            print(f"❌ Error storing ideal contract template: {e}")
            raise
    
    def get_ideal_contract(self, template_id: str) -> Optional[Dict]:
        """
        Retrieve an ideal contract template by ID.
        
        Args:
            template_id: The template ID
            
        Returns:
            Template data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[template_id],
                include=["documents", "metadatas"]
            )
            
            if results['ids']:
                document_text = results['documents'][0]
                template_data = json.loads(document_text)
                return template_data
            
        except Exception as e:
            print(f"⚠️ Error retrieving ideal contract template: {e}")
        
        return None
    
    def list_ideal_contracts(
        self, 
        category: Optional[ContractCategory] = None
    ) -> List[Dict]:
        """
        List all ideal contract templates, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template metadata
        """
        try:
            where_filter = {}
            if category:
                where_filter["category"] = category.value
            
            results = self.collection.get(
                where=where_filter if where_filter else None,
                include=["metadatas"]
            )
            
            return results['metadatas'] or []
            
        except Exception as e:
            print(f"⚠️ Error listing ideal contracts: {e}")
            return []
    
    def search_similar_templates(
        self,
        query_embedding: np.ndarray,
        category: Optional[ContractCategory] = None,
        n_results: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar ideal contract templates using embeddings.
        
        Args:
            query_embedding: Query embedding vector
            category: Optional category filter
            n_results: Number of results to return
            
        Returns:
            List of (template_data, similarity_score) tuples
        """
        where_filter = {}
        if category:
            where_filter["category"] = category.value
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            templates_with_scores = []
            for i in range(len(results['ids'][0])):
                document_text = results['documents'][0][i]
                template_data = json.loads(document_text)
                # ChromaDB returns distances, convert to similarity
                similarity = 1.0 - results['distances'][0][i]
                templates_with_scores.append((template_data, similarity))
            
            return templates_with_scores
            
        except Exception as e:
            print(f"⚠️ Error searching similar templates: {e}")
            return []
    
    def delete_ideal_contract(self, template_id: str) -> bool:
        """
        Delete an ideal contract template.
        
        Args:
            template_id: The template ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Check if template exists
            existing = self.get_ideal_contract(template_id)
            if not existing:
                print(f"⚠️ Template not found: {template_id}")
                return False
            
            # Delete from ChromaDB
            self.collection.delete(ids=[template_id])
            print(f"🗑️ Deleted ideal contract template: {template_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting ideal contract template: {e}")
            return False
    
    def update_ideal_contract(
        self,
        template_id: str,
        updates: Dict,
        new_embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Update an existing ideal contract template.
        
        Args:
            template_id: Template ID to update
            updates: Dictionary of fields to update
            new_embedding: Optional new embedding vector
            
        Returns:
            True if updated successfully
        """
        try:
            # Get existing template
            existing_template = self.get_ideal_contract(template_id)
            if not existing_template:
                print(f"⚠️ Template not found for update: {template_id}")
                return False
            
            # Merge updates
            existing_template.update(updates)
            existing_template["metadata"]["updated_timestamp"] = datetime.now().isoformat()
            
            # Delete old version
            self.collection.delete(ids=[template_id])
            
            # Store updated version
            document_text = json.dumps(existing_template, indent=2)
            embedding_to_use = new_embedding if new_embedding is not None else np.zeros(384)  # Default embedding size
            
            self.collection.add(
                ids=[template_id],
                documents=[document_text],
                metadatas=[existing_template["metadata"]],
                embeddings=[embedding_to_use.tolist()]
            )
            
            print(f"✅ Updated ideal contract template: {template_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating ideal contract template: {e}")
            return False


def create_sample_rental_template() -> Dict:
    """Create a sample rental contract template for testing."""
    return {
        "category": ContractCategory.RENTAL,
        "title": "Standard Residential Rental Agreement",
        "description": "Comprehensive rental agreement template with essential tenant protections",
        "essential_clauses": [
            {
                "name": "rent_amount",
                "description": "Monthly rent amount must be clearly specified",
                "importance": 10,
                "keywords": ["rent", "monthly payment", "amount", "due"],
                "required": True
            },
            {
                "name": "security_deposit",
                "description": "Security deposit terms and return conditions",
                "importance": 9,
                "keywords": ["security deposit", "damage deposit", "refund"],
                "required": True
            },
            {
                "name": "lease_term",
                "description": "Start and end dates of the lease",
                "importance": 10,
                "keywords": ["lease term", "start date", "end date", "duration"],
                "required": True
            },
            {
                "name": "property_description",
                "description": "Clear description of the rental property",
                "importance": 8,
                "keywords": ["property", "address", "premises", "unit"],
                "required": True
            },
            {
                "name": "maintenance_responsibilities",
                "description": "Who is responsible for maintenance and repairs",
                "importance": 7,
                "keywords": ["maintenance", "repairs", "responsibility"],
                "required": True
            }
        ],
        "risk_factors": [
            {
                "name": "automatic_renewal",
                "description": "Automatic renewal without proper notice period",
                "risk_level": RiskLevel.HIGH.value,
                "keywords": ["automatic renewal", "auto-renew", "automatically extend"],
                "penalty_score": -15
            },
            {
                "name": "excessive_late_fees",
                "description": "Late fees exceeding reasonable amounts",
                "risk_level": RiskLevel.MEDIUM.value,
                "keywords": ["late fee", "penalty", "overdue"],
                "penalty_score": -10
            },
            {
                "name": "no_deposit_return_clause",
                "description": "Missing or unclear security deposit return terms",
                "risk_level": RiskLevel.HIGH.value,
                "keywords": ["deposit return", "refund conditions"],
                "penalty_score": -20
            }
        ],
        "compliance_requirements": [
            {
                "name": "fair_housing",
                "description": "Must comply with Fair Housing Act",
                "required": True,
                "keywords": ["discrimination", "equal opportunity", "fair housing"]
            },
            {
                "name": "local_rent_control",
                "description": "Must comply with local rent control laws",
                "required": True,
                "keywords": ["rent control", "rent stabilization"]
            }
        ],
        "scoring_weights": {
            "essential_clauses": 0.6,  # 60% of score
            "risk_factors": 0.3,       # 30% of score  
            "compliance": 0.1          # 10% of score
        }
    }


def create_sample_employment_template() -> Dict:
    """Create a sample employment contract template for testing."""
    return {
        "category": ContractCategory.EMPLOYMENT,
        "title": "Standard Employment Agreement",
        "description": "Comprehensive employment contract with worker protections",
        "essential_clauses": [
            {
                "name": "job_title_duties",
                "description": "Clear job title and description of duties",
                "importance": 9,
                "keywords": ["job title", "position", "duties", "responsibilities"],
                "required": True
            },
            {
                "name": "compensation",
                "description": "Salary, hourly rate, or compensation structure",
                "importance": 10,
                "keywords": ["salary", "wage", "compensation", "pay"],
                "required": True
            },
            {
                "name": "work_schedule",
                "description": "Working hours and schedule expectations",
                "importance": 8,
                "keywords": ["hours", "schedule", "workweek", "overtime"],
                "required": True
            },
            {
                "name": "benefits",
                "description": "Health insurance, vacation, and other benefits",
                "importance": 7,
                "keywords": ["benefits", "insurance", "vacation", "sick leave"],
                "required": False
            },
            {
                "name": "termination_clause",
                "description": "Conditions for employment termination",
                "importance": 9,
                "keywords": ["termination", "resignation", "notice period"],
                "required": True
            }
        ],
        "risk_factors": [
            {
                "name": "unpaid_overtime",
                "description": "No overtime compensation for eligible employees",
                "risk_level": RiskLevel.HIGH.value,
                "keywords": ["overtime", "unpaid", "exempt"],
                "penalty_score": -20
            },
            {
                "name": "broad_non_compete",
                "description": "Overly broad or unreasonable non-compete clauses",
                "risk_level": RiskLevel.MEDIUM.value,
                "keywords": ["non-compete", "non-competition", "restraint"],
                "penalty_score": -15
            },
            {
                "name": "at_will_without_protection",
                "description": "At-will employment without wrongful termination protections",
                "risk_level": RiskLevel.MEDIUM.value,
                "keywords": ["at-will", "terminate without cause"],
                "penalty_score": -10
            }
        ],
        "compliance_requirements": [
            {
                "name": "wage_hour_compliance",
                "description": "Must comply with federal and state wage and hour laws",
                "required": True,
                "keywords": ["minimum wage", "overtime", "FLSA"]
            },
            {
                "name": "equal_employment",
                "description": "Must comply with equal employment opportunity laws",
                "required": True,
                "keywords": ["EEO", "discrimination", "equal opportunity"]
            }
        ],
        "scoring_weights": {
            "essential_clauses": 0.5,
            "risk_factors": 0.4,
            "compliance": 0.1
        }
    }