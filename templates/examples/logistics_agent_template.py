"""
Example: Logistics Agent Template

This demonstrates how to create a logistics/supply chain agent using the template framework.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.graph_management import GraphManager
from templates.prediction_framework import PredictionFramework
from templates.statistical_methods import StatisticalMethods
from templates.llm_integration import LLMIntegration, create_default_schema


class LogisticsAgentTemplate(BaseSpecializedAgent):
    """
    Example logistics agent using the template framework.
    
    This agent can reason about supply chain relationships, predict delivery times,
    optimize routes, and generate logistics strategies.
    """

    def __init__(
        self,
        supply_chain_nodes: Optional[List[str]] = None,
        supply_chain_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "logistics-agent",
        model_name: str = "gpt-4o",
        **kwargs,
    ):
        """
        Initialize the logistics agent.
        
        Args:
            supply_chain_nodes: List of supply chain node names (e.g., ['warehouse', 'transport', 'delivery'])
            supply_chain_edges: List of (source, target) relationship tuples
            max_loops: Maximum reasoning loops
            agent_name: Agent identifier
            model_name: LLM model name
            **kwargs: Additional arguments
        """
        super().__init__(
            max_loops=max_loops,
            agent_name=agent_name,
            agent_description="Logistics and Supply Chain Analysis Agent",
            model_name=model_name,
            **kwargs,
        )

        # Initialize graph manager for supply chain
        self.graph_manager = GraphManager(graph_type="supply_chain")
        
        # Initialize prediction framework
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager,
            use_nonlinear=False,  # Logistics often linear
        )
        
        # Initialize statistical methods
        self.statistical_methods = StatisticalMethods(
            graph_manager=self.graph_manager,
            prediction_framework=self.prediction_framework,
            standardization_stats=self.prediction_framework.standardization_stats,
        )
        
        # Initialize LLM integration
        self.llm_integration = LLMIntegration(
            agent=self,
            max_loops=self.domain_max_loops if isinstance(self.domain_max_loops, int) else 3,
        )
        
        # Add supply chain nodes and edges
        if supply_chain_nodes:
            self.graph_manager.add_nodes_from(supply_chain_nodes)
        
        if supply_chain_edges:
            for source, target in supply_chain_edges:
                self.add_supply_chain_relationship(source, target)

    def _get_domain_schema(self) -> Optional[Dict[str, Any]]:
        """Return the logistics analysis schema."""
        return create_default_schema(
            function_name="generate_logistics_analysis",
            description="Generates logistics analysis and optimization recommendations",
            properties={
                "supply_chain_analysis": {
                    "type": "string",
                    "description": "Analysis of supply chain efficiency and bottlenecks"
                },
                "route_optimization": {
                    "type": "string",
                    "description": "Route optimization recommendations"
                },
                "delivery_scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scenario_name": {"type": "string"},
                            "interventions": {"type": "object"},
                            "expected_delivery_times": {"type": "object"},
                            "cost_impact": {"type": "object"}
                        }
                    },
                    "description": "Different delivery scenarios and their impacts"
                },
                "optimization_recommendations": {
                    "type": "string",
                    "description": "Recommendations for supply chain optimization"
                }
            },
            required=[
                "supply_chain_analysis",
                "route_optimization",
                "delivery_scenarios",
                "optimization_recommendations"
            ]
        )

    def _build_domain_prompt(self, task: str) -> str:
        """Build a logistics analysis prompt."""
        return (
            f"You are a Logistics and Supply Chain Analysis Agent.\n"
            f"Task: {task}\n"
            f"Current supply chain model has {len(self.graph_manager.get_nodes())} nodes and "
            f"{len(self.graph_manager.get_edges())} relationships.\n"
            f"Analyze the supply chain and provide optimization recommendations.\n"
        )

    def _domain_specific_setup(self) -> None:
        """Set up logistics-specific attributes."""
        # Logistics-specific setup (e.g., capacity constraints, service levels)
        self.capacity_constraints: Dict[str, float] = {}
        self.service_level_targets: Dict[str, float] = {}

    def add_supply_chain_relationship(
        self,
        source: str,
        target: str,
        lead_time: float = 0.0,
        capacity: float = 1.0,
        confidence: float = 1.0
    ) -> None:
        """Add a supply chain relationship."""
        self.graph_manager.add_relationship(
            source=source,
            target=target,
            strength=lead_time,
            confidence=confidence,
            capacity=capacity,
        )

    def predict_delivery_outcomes(
        self,
        current_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict delivery outcomes given current state and interventions."""
        return self.prediction_framework.predict_outcomes(
            current_state, interventions
        )

    def generate_delivery_scenarios(
        self,
        current_state: Dict[str, float],
        target_nodes: List[str],
        max_scenarios: int = 5
    ) -> List:
        """Generate delivery scenarios."""
        return self.prediction_framework.generate_counterfactual_scenarios(
            current_state, target_nodes, max_scenarios
        )

    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        supply_chain_state: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run logistics analysis.
        
        Args:
            task: Logistics task or question
            supply_chain_state: Current supply chain state dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with logistics analysis results
        """
        if task and isinstance(task, str) and supply_chain_state is None:
            return self.llm_integration.run_llm_domain_analysis(
                task,
                build_prompt_fn=self._build_domain_prompt,
            )
        
        # Supply chain state analysis
        if supply_chain_state:
            return {
                "supply_chain_state": supply_chain_state,
                "predictions": self.prediction_framework.predict_outcomes(supply_chain_state, {}),
                "scenarios": self.generate_delivery_scenarios(supply_chain_state, list(supply_chain_state.keys())),
            }
        
        return {"error": "Either task or supply_chain_state must be provided"}

