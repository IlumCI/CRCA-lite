"""
Example: Causal Agent Template

This demonstrates how to create a causal reasoning agent using the template framework.
This is a simplified example showing the pattern - the full CRCA.py can be refactored
to use this pattern.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.graph_management import GraphManager
from templates.prediction_framework import PredictionFramework, CounterfactualScenario
from templates.statistical_methods import StatisticalMethods
from templates.llm_integration import LLMIntegration, create_default_schema


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


class CausalAgentTemplate(BaseSpecializedAgent):
    """
    Example causal reasoning agent using the template framework.
    
    This demonstrates the pattern for creating specialized agents:
    1. Inherit from BaseSpecializedAgent
    2. Use composition with extracted modules
    3. Implement domain-specific methods
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "causal-agent",
        agent_description: str = "Causal Reasoning Agent",
        model_name: str = "gpt-4o",
        use_nonlinear: bool = True,
        **kwargs,
    ):
        """
        Initialize the causal agent.
        
        Args:
            variables: List of variable names
            causal_edges: List of (source, target) edge tuples
            max_loops: Maximum reasoning loops
            agent_name: Agent identifier
            agent_description: Agent description
            model_name: LLM model name
            use_nonlinear: Whether to use nonlinear prediction
            **kwargs: Additional arguments
        """
        # Initialize base class
        super().__init__(
            max_loops=max_loops,
            agent_name=agent_name,
            agent_description=agent_description,
            model_name=model_name,
            **kwargs,
        )

        # Initialize graph manager
        self.graph_manager = GraphManager(graph_type="causal")
        
        # Initialize prediction framework
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager,
            use_nonlinear=use_nonlinear,
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
        
        # Add variables and edges
        if variables:
            self.graph_manager.add_nodes_from(variables)
        
        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

    def _get_domain_schema(self) -> Optional[Dict[str, Any]]:
        """Return the causal analysis schema."""
        return create_default_schema(
            function_name="generate_causal_analysis",
            description="Generates structured causal reasoning and counterfactual analysis",
            properties={
                "causal_analysis": {
                    "type": "string",
                    "description": "Analysis of causal relationships and mechanisms"
                },
                "intervention_planning": {
                    "type": "string",
                    "description": "Planned interventions to test causal hypotheses"
                },
                "counterfactual_scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scenario_name": {"type": "string"},
                            "interventions": {"type": "object"},
                            "expected_outcomes": {"type": "object"},
                            "reasoning": {"type": "string"}
                        }
                    },
                    "description": "Multiple counterfactual scenarios to explore"
                },
                "causal_strength_assessment": {
                    "type": "string",
                    "description": "Assessment of causal relationship strengths and confounders"
                },
                "optimal_solution": {
                    "type": "string",
                    "description": "Recommended optimal solution based on causal analysis"
                }
            },
            required=[
                "causal_analysis",
                "intervention_planning",
                "counterfactual_scenarios",
                "causal_strength_assessment",
                "optimal_solution"
            ]
        )

    def _build_domain_prompt(self, task: str) -> str:
        """Build a causal reasoning prompt."""
        return (
            f"You are a Causal Reasoning with Counterfactual Analysis (CR-CA) agent.\n"
            f"Problem: {task}\n"
            f"Current causal graph has {len(self.graph_manager.get_nodes())} variables and "
            f"{len(self.graph_manager.get_edges())} relationships.\n"
        )

    def _domain_specific_setup(self) -> None:
        """Set up domain-specific attributes."""
        # Causal-specific setup can go here
        pass

    def add_causal_relationship(
        self,
        source: str,
        target: str,
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        """Add a causal relationship."""
        self.graph_manager.add_relationship(
            source=source,
            target=target,
            strength=strength,
            relation_type=relation_type,
            confidence=confidence,
        )

    def predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict outcomes given state and interventions."""
        return self.prediction_framework.predict_outcomes(factual_state, interventions)

    def generate_counterfactual_scenarios(
        self,
        factual_state: Dict[str, float],
        target_variables: List[str],
        max_scenarios: int = 5
    ) -> List[CounterfactualScenario]:
        """Generate counterfactual scenarios."""
        return self.prediction_framework.generate_counterfactual_scenarios(
            factual_state, target_variables, max_scenarios
        )

    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the agent with either LLM analysis or state evolution.
        
        Args:
            task: Task string for LLM analysis, or state dict
            initial_state: Initial state dictionary
            target_variables: Target variables for counterfactuals
            max_steps: Maximum evolution steps
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        # LLM-based analysis
        if task is not None and isinstance(task, str) and initial_state is None and not task.strip().startswith('{'):
            return self.llm_integration.run_llm_domain_analysis(
                task,
                build_prompt_fn=self._build_domain_prompt,
                post_process_fn=self._post_process_llm_results,
            )

        # State evolution
        if task is not None and initial_state is None:
            initial_state = task

        if not isinstance(initial_state, dict):
            try:
                import json
                parsed = json.loads(initial_state)
                if isinstance(parsed, dict):
                    initial_state = parsed
                else:
                    return {"error": "initial_state JSON must decode to a dict"}
            except Exception:
                return {"error": "initial_state must be a dict or JSON-encoded dict"}

        if target_variables is None:
            target_variables = self.graph_manager.get_nodes()

        def _resolve_max_steps(value: Union[int, str]) -> int:
            if isinstance(value, str) and value == "auto":
                return max(1, len(self.graph_manager.get_nodes()))
            try:
                return int(value)
            except Exception:
                return max(1, len(self.graph_manager.get_nodes()))

        effective_steps = _resolve_max_steps(max_steps)
        current_state = initial_state.copy()
        for step in range(effective_steps):
            current_state = self.prediction_framework.predict_outcomes(current_state, {})

        self.prediction_framework.ensure_standardization_stats(current_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            current_state,
            target_variables,
            max_scenarios=5
        )

        return {
            "initial_state": initial_state,
            "evolved_state": current_state,
            "counterfactual_scenarios": counterfactual_scenarios,
            "graph_info": {
                "nodes": self.graph_manager.get_nodes(),
                "edges": self.graph_manager.get_edges(),
                "is_dag": self.graph_manager.is_dag()
            },
            "steps": effective_steps
        }

    def _post_process_llm_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process LLM analysis results."""
        default_state = {var: 0.0 for var in self.graph_manager.get_nodes()}
        self.prediction_framework.ensure_standardization_stats(default_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            default_state,
            self.graph_manager.get_nodes()[:5],
            max_scenarios=5
        )
        results['counterfactual_scenarios'] = counterfactual_scenarios
        results['graph_info'] = {
            'nodes': self.graph_manager.get_nodes(),
            'edges': self.graph_manager.get_edges(),
            'is_dag': self.graph_manager.is_dag()
        }
        return results

    # Convenience methods that delegate to modules
    def get_nodes(self) -> List[str]:
        """Get all nodes."""
        return self.graph_manager.get_nodes()

    def get_edges(self) -> List[Tuple[str, str]]:
        """Get all edges."""
        return self.graph_manager.get_edges()

    def is_dag(self) -> bool:
        """Check if graph is a DAG."""
        return self.graph_manager.is_dag()

    def fit_from_dataframe(self, df: Any, variables: List[str], **kwargs) -> None:
        """Fit model from dataframe."""
        self.statistical_methods.fit_from_dataframe(df, variables, **kwargs)
        # Update prediction framework stats
        self.prediction_framework.standardization_stats = self.statistical_methods.standardization_stats

