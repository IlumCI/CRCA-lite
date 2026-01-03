"""
Mandate Generator: Convert CorporateProposal to bolt.diy execution mandate.

This module provides utilities to convert corporate governance proposals
into structured execution mandates for autonomous code generation.
"""

from typing import Dict, Any, Optional, List
import time
import uuid
from dataclasses import dataclass

# Type hints for CorporateProposal (avoiding circular import)
# In practice, this would import from corposwarm.py
# from branches.crca_cg.corposwarm import CorporateProposal, ProposalType, CorporateSwarm


def proposal_to_mandate(
    proposal: Any,  # CorporateProposal
    corporate_swarm: Optional[Any] = None,  # CorporateSwarm
    mandate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert corporate proposal to execution mandate.
    
    Extracts:
    - Objectives from proposal description
    - Budget from proposal budget_impact
    - Constraints based on proposal type
    - Governance metadata (proposal_id, ESG requirements, risk thresholds)
    - Deployment config if proposal includes deployment
    
    Args:
        proposal: CorporateProposal instance
        corporate_swarm: Optional CorporateSwarm instance for ESG/risk context
        mandate_id: Optional custom mandate ID (defaults to proposal_id)
        
    Returns:
        Dict matching bolt.diy Mandate type structure
    """
    # Generate mandate ID
    mandate_id = mandate_id or proposal.proposal_id or str(uuid.uuid4())
    
    # Extract objectives from proposal description
    objectives = _extract_objectives(proposal)
    
    # Determine constraints based on proposal type
    constraints = _determine_constraints(proposal)
    
    # Calculate budget from proposal budget_impact
    budget = _calculate_budget(proposal)
    
    # Extract deliverables
    deliverables = _extract_deliverables(proposal)
    
    # Build governance metadata
    governance = _build_governance_metadata(proposal, corporate_swarm)
    
    # Determine iteration config
    iteration_config = _determine_iteration_config(proposal)
    
    # Determine deployment config
    deployment = _determine_deployment_config(proposal)
    
    # Build mandate
    mandate = {
        "mandate_id": mandate_id,
        "objectives": objectives,
        "constraints": constraints,
        "budget": budget,
        "deliverables": deliverables,
        "governance": governance,
        "iteration_config": iteration_config,
    }
    
    if deployment:
        mandate["deployment"] = deployment
    
    return mandate


def _extract_objectives(proposal: Any) -> List[str]:
    """Extract objectives from proposal description."""
    objectives = []
    
    # Use title as primary objective
    if proposal.title:
        objectives.append(proposal.title)
    
    # Parse description for additional objectives
    if proposal.description:
        # Split by common separators (newlines, periods, semicolons)
        desc_lines = proposal.description.replace('\n', '. ').split('. ')
        for line in desc_lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short fragments
                objectives.append(line)
    
    # If no objectives found, create default
    if not objectives:
        objectives.append(f"Execute {proposal.proposal_type.value if hasattr(proposal.proposal_type, 'value') else str(proposal.proposal_type)}")
    
    return objectives[:5]  # Limit to 5 objectives


def _determine_constraints(proposal: Any) -> Dict[str, Any]:
    """Determine execution constraints based on proposal type."""
    # Default constraints
    constraints = {
        "language": "ts",  # Default to TypeScript
        "maxDependencies": 50,
        "noNetwork": False,
        "allowedPackages": [],
        "maxFileSize": 100000,  # 100KB
        "maxFiles": 100,
    }
    
    # Adjust based on proposal type
    proposal_type = str(proposal.proposal_type) if hasattr(proposal, 'proposal_type') else ""
    
    # Product launches might need more dependencies
    if "product_launch" in proposal_type.lower():
        constraints["maxDependencies"] = 100
        constraints["maxFiles"] = 200
    
    # Compliance updates might need stricter constraints
    if "compliance" in proposal_type.lower():
        constraints["noNetwork"] = True
        constraints["maxDependencies"] = 20
    
    # Strategic initiatives might need more flexibility
    if "strategic" in proposal_type.lower():
        constraints["maxDependencies"] = 150
        constraints["maxFiles"] = 300
    
    # Check metadata for custom constraints
    if hasattr(proposal, 'metadata') and proposal.metadata:
        if 'language' in proposal.metadata:
            constraints["language"] = proposal.metadata['language']
        if 'maxDependencies' in proposal.metadata:
            constraints["maxDependencies"] = int(proposal.metadata['maxDependencies'])
        if 'noNetwork' in proposal.metadata:
            constraints["noNetwork"] = bool(proposal.metadata['noNetwork'])
        if 'allowedPackages' in proposal.metadata:
            constraints["allowedPackages"] = proposal.metadata['allowedPackages']
    
    return constraints


def _calculate_budget(proposal: Any) -> Dict[str, Any]:
    """Calculate execution budget from proposal budget_impact."""
    # Base budget calculation
    # Assume budget_impact is in dollars
    budget_impact = float(proposal.budget_impact) if hasattr(proposal, 'budget_impact') else 100.0
    
    # Convert to token/time/cost budget
    # Rough estimates:
    # - $1 = ~10,000 tokens
    # - $1 = ~60 seconds execution time
    # - Cost is the budget_impact itself
    
    token_budget = int(budget_impact * 10000)
    time_budget = int(budget_impact * 60)  # seconds
    cost_budget = budget_impact
    
    # Ensure minimums
    token_budget = max(token_budget, 10000)  # At least 10k tokens
    time_budget = max(time_budget, 60)  # At least 60 seconds
    cost_budget = max(cost_budget, 1.0)  # At least $1
    
    return {
        "token": token_budget,
        "time": time_budget,
        "cost": cost_budget,
    }


def _extract_deliverables(proposal: Any) -> List[str]:
    """Extract deliverables from proposal."""
    deliverables = []
    
    # Check metadata for explicit deliverables
    if hasattr(proposal, 'metadata') and proposal.metadata:
        if 'deliverables' in proposal.metadata:
            if isinstance(proposal.metadata['deliverables'], list):
                deliverables.extend(proposal.metadata['deliverables'])
            elif isinstance(proposal.metadata['deliverables'], str):
                deliverables.append(proposal.metadata['deliverables'])
    
    # If no explicit deliverables, infer from proposal type
    if not deliverables:
        proposal_type = str(proposal.proposal_type) if hasattr(proposal, 'proposal_type') else ""
        
        if "product" in proposal_type.lower():
            deliverables = ["src/App.tsx", "package.json", "README.md"]
        elif "website" in proposal_type.lower() or "web" in proposal_type.lower():
            deliverables = ["index.html", "src/index.ts", "package.json"]
        else:
            deliverables = ["src/index.ts", "package.json"]
    
    return deliverables


def _build_governance_metadata(
    proposal: Any,
    corporate_swarm: Optional[Any] = None
) -> Dict[str, Any]:
    """Build governance metadata from proposal and corporate swarm context."""
    governance: Dict[str, Any] = {
        "proposal_id": proposal.proposal_id if hasattr(proposal, 'proposal_id') else "",
    }
    
    # Add ESG requirements if available from corporate swarm
    if corporate_swarm and hasattr(corporate_swarm, 'calculate_esg_score'):
        try:
            esg_score = corporate_swarm.calculate_esg_score()
            governance["esg_requirements"] = {
                "environmental_score": float(esg_score.environmental_score) if hasattr(esg_score, 'environmental_score') else 0.0,
                "social_score": float(esg_score.social_score) if hasattr(esg_score, 'social_score') else 0.0,
                "governance_score": float(esg_score.governance_score) if hasattr(esg_score, 'governance_score') else 0.0,
                "overall_score": float(esg_score.overall_score) if hasattr(esg_score, 'overall_score') else 0.0,
            }
        except Exception:
            pass  # ESG calculation failed, skip
    
    # Add risk threshold based on proposal type
    proposal_type = str(proposal.proposal_type) if hasattr(proposal, 'proposal_type') else ""
    
    # Higher risk threshold for critical proposals
    if "compliance" in proposal_type.lower() or "audit" in proposal_type.lower():
        governance["risk_threshold"] = 30.0  # Low risk tolerance
    elif "strategic" in proposal_type.lower() or "merger" in proposal_type.lower():
        governance["risk_threshold"] = 70.0  # Higher risk tolerance
    else:
        governance["risk_threshold"] = 50.0  # Default
    
    # Add approval chain if specified
    if hasattr(proposal, 'metadata') and proposal.metadata:
        if 'approval_chain' in proposal.metadata:
            governance["approval_chain"] = proposal.metadata['approval_chain']
    
    # Add causal analysis requirement if proposal has causal analysis
    if hasattr(proposal, 'causal_analysis') and proposal.causal_analysis:
        governance["causal_analysis_required"] = True
    
    return governance


def _determine_iteration_config(proposal: Any) -> Dict[str, Any]:
    """Determine iteration configuration based on proposal complexity."""
    # Default config
    config = {
        "max_iterations": 3,
        "test_required": True,
        "quality_threshold": 0.7,
    }
    
    # Adjust based on proposal type
    proposal_type = str(proposal.proposal_type) if hasattr(proposal, 'proposal_type') else ""
    
    # More iterations for complex proposals
    if "strategic" in proposal_type.lower() or "product" in proposal_type.lower():
        config["max_iterations"] = 5
        config["quality_threshold"] = 0.8
    
    # Fewer iterations for simple proposals
    if "compliance" in proposal_type.lower() or "policy" in proposal_type.lower():
        config["max_iterations"] = 2
        config["quality_threshold"] = 0.6
    
    # Check metadata for custom config
    if hasattr(proposal, 'metadata') and proposal.metadata:
        if 'max_iterations' in proposal.metadata:
            config["max_iterations"] = int(proposal.metadata['max_iterations'])
        if 'test_required' in proposal.metadata:
            config["test_required"] = bool(proposal.metadata['test_required'])
        if 'quality_threshold' in proposal.metadata:
            config["quality_threshold"] = float(proposal.metadata['quality_threshold'])
    
    return config


def _determine_deployment_config(proposal: Any) -> Optional[Dict[str, Any]]:
    """Determine deployment configuration if proposal requires deployment."""
    # Check if deployment is requested
    if hasattr(proposal, 'metadata') and proposal.metadata:
        if 'deployment' in proposal.metadata:
            deployment_config = proposal.metadata['deployment']
            if isinstance(deployment_config, dict):
                return {
                    "enabled": True,
                    "provider": deployment_config.get('provider', 'netlify'),
                    "target": deployment_config.get('target'),
                    "auto_deploy": deployment_config.get('auto_deploy', True),
                }
            elif deployment_config is True:
                # Default deployment config
                return {
                    "enabled": True,
                    "provider": "netlify",
                    "auto_deploy": True,
                }
    
    # Check proposal type for deployment hints
    proposal_type = str(proposal.proposal_type) if hasattr(proposal, 'proposal_type') else ""
    
    # Product launches typically need deployment
    if "product" in proposal_type.lower() or "launch" in proposal_type.lower():
        return {
            "enabled": True,
            "provider": "netlify",
            "auto_deploy": True,
        }
    
    return None


def create_mandate_from_proposal(
    proposal: Any,
    corporate_swarm: Optional[Any] = None,
    mandate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to create a mandate from a proposal.
    
    This is the main entry point for converting proposals to mandates.
    
    Args:
        proposal: CorporateProposal instance
        corporate_swarm: Optional CorporateSwarm instance
        mandate_id: Optional custom mandate ID
        
    Returns:
        Dict matching bolt.diy Mandate type structure
    """
    return proposal_to_mandate(proposal, corporate_swarm, mandate_id)

