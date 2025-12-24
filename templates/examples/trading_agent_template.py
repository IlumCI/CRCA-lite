"""
Example: Trading Agent Template

This demonstrates how to create a trading/financial agent using the template framework.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.graph_management import GraphManager
from templates.prediction_framework import PredictionFramework
from templates.statistical_methods import StatisticalMethods
from templates.llm_integration import LLMIntegration, create_default_schema


class TradingAgentTemplate(BaseSpecializedAgent):
    """
    Example trading agent using the template framework.
    
    This agent can reason about market relationships, predict price movements,
    and generate trading strategies.
    """

    def __init__(
        self,
        market_variables: Optional[List[str]] = None,
        market_relationships: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "trading-agent",
        model_name: str = "gpt-4o",
        **kwargs,
    ):
        """
        Initialize the trading agent.
        
        Args:
            market_variables: List of market variable names (e.g., ['SPY', 'VIX', 'USD'])
            market_relationships: List of (source, target) relationship tuples
            max_loops: Maximum reasoning loops
            agent_name: Agent identifier
            model_name: LLM model name
            **kwargs: Additional arguments
        """
        super().__init__(
            max_loops=max_loops,
            agent_name=agent_name,
            agent_description="Trading and Financial Analysis Agent",
            model_name=model_name,
            **kwargs,
        )

        # Initialize graph manager for market relationships
        self.graph_manager = GraphManager(graph_type="market")
        
        # Initialize prediction framework
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager,
            use_nonlinear=True,  # Markets are nonlinear
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
        
        # Add market variables and relationships
        if market_variables:
            self.graph_manager.add_nodes_from(market_variables)
        
        if market_relationships:
            for source, target in market_relationships:
                self.add_market_relationship(source, target)

    def _get_domain_schema(self) -> Optional[Dict[str, Any]]:
        """Return the trading analysis schema."""
        return create_default_schema(
            function_name="generate_trading_analysis",
            description="Generates trading analysis and strategy recommendations",
            properties={
                "market_analysis": {
                    "type": "string",
                    "description": "Analysis of current market conditions and trends"
                },
                "risk_assessment": {
                    "type": "string",
                    "description": "Assessment of market risks and volatility"
                },
                "trading_strategies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "strategy_name": {"type": "string"},
                            "entry_signals": {"type": "object"},
                            "exit_signals": {"type": "object"},
                            "risk_parameters": {"type": "object"},
                            "expected_returns": {"type": "object"}
                        }
                    },
                    "description": "Recommended trading strategies"
                },
                "portfolio_recommendations": {
                    "type": "string",
                    "description": "Portfolio allocation recommendations"
                }
            },
            required=[
                "market_analysis",
                "risk_assessment",
                "trading_strategies",
                "portfolio_recommendations"
            ]
        )

    def _build_domain_prompt(self, task: str) -> str:
        """Build a trading analysis prompt."""
        return (
            f"You are a Trading and Financial Analysis Agent.\n"
            f"Task: {task}\n"
            f"Current market model has {len(self.graph_manager.get_nodes())} variables and "
            f"{len(self.graph_manager.get_edges())} relationships.\n"
            f"Analyze the market conditions and provide trading recommendations.\n"
        )

    def _domain_specific_setup(self) -> None:
        """Set up trading-specific attributes."""
        # Trading-specific setup (e.g., risk limits, position sizing)
        self.risk_limits: Dict[str, float] = {}
        self.position_sizing_rules: Dict[str, Any] = {}

    def add_market_relationship(
        self,
        source: str,
        target: str,
        correlation: float = 0.0,
        confidence: float = 1.0
    ) -> None:
        """Add a market relationship (e.g., correlation, lead-lag)."""
        self.graph_manager.add_relationship(
            source=source,
            target=target,
            strength=correlation,
            confidence=confidence,
        )

    def predict_market_outcomes(
        self,
        current_market_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict market outcomes given current state and interventions."""
        return self.prediction_framework.predict_outcomes(
            current_market_state, interventions
        )

    def generate_trading_scenarios(
        self,
        current_state: Dict[str, float],
        target_assets: List[str],
        max_scenarios: int = 5
    ) -> List:
        """Generate trading scenarios."""
        return self.prediction_framework.generate_counterfactual_scenarios(
            current_state, target_assets, max_scenarios
        )

    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        market_state: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run trading analysis.
        
        Args:
            task: Trading task or question
            market_state: Current market state dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with trading analysis results
        """
        if task and isinstance(task, str) and market_state is None:
            return self.llm_integration.run_llm_domain_analysis(
                task,
                build_prompt_fn=self._build_domain_prompt,
            )
        
        # Market state analysis
        if market_state:
            return {
                "market_state": market_state,
                "predictions": self.prediction_framework.predict_outcomes(market_state, {}),
                "scenarios": self.generate_trading_scenarios(market_state, list(market_state.keys())),
            }
        
        return {"error": "Either task or market_state must be provided"}

