"""
Drag-and-Drop Feature Composition Examples

Demonstrates how to use the template framework modules as "drag-and-drop" features
to quickly compose specialized agents.
"""

from typing import Dict, Any, List, Optional
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import (
    GraphFeatureMixin,
    PredictionFeatureMixin,
    StatisticsFeatureMixin,
    LLMFeatureMixin,
    FullFeatureMixin
)
from templates.module_registry import compose_agent, ModuleRegistry
from templates.llm_integration import create_default_schema


# ============================================================================
# Example 1: Minimal Agent (LLM only)
# ============================================================================

class MinimalAgent(BaseSpecializedAgent, LLMFeatureMixin):
    """Minimal agent with just LLM capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_llm_feature()
    
    def _get_domain_schema(self):
        return create_default_schema(
            function_name="minimal_analysis",
            description="Minimal analysis",
            properties={"result": {"type": "string"}}
        )
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Analyze: {task}"
    
    def _domain_specific_setup(self):
        pass


# ============================================================================
# Example 2: Graph + LLM Agent
# ============================================================================

class GraphLLMAgent(BaseSpecializedAgent, GraphFeatureMixin, LLMFeatureMixin):
    """Agent with graph management and LLM capabilities."""
    
    def __init__(self, variables=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.init_graph_feature(variables=variables, edges=edges)
        self.init_llm_feature()
    
    def _get_domain_schema(self):
        return create_default_schema(
            function_name="graph_analysis",
            description="Graph-based analysis",
            properties={
                "analysis": {"type": "string"},
                "graph_info": {"type": "object"}
            }
        )
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Task: {task}\nGraph has {len(self.get_nodes())} nodes."
    
    def _domain_specific_setup(self):
        pass


# ============================================================================
# Example 3: Full-Featured Agent (All Features)
# ============================================================================

class FullFeaturedAgent(BaseSpecializedAgent, FullFeatureMixin):
    """Agent with all features - graph, prediction, statistics, LLM."""
    
    def __init__(self, variables=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.init_all_features(
            variables=variables,
            edges=edges,
            use_nonlinear=True
        )
    
    def _get_domain_schema(self):
        return create_default_schema(
            function_name="full_analysis",
            description="Full-featured analysis",
            properties={
                "analysis": {"type": "string"},
                "predictions": {"type": "object"},
                "statistics": {"type": "object"}
            }
        )
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Full analysis: {task}"
    
    def _domain_specific_setup(self):
        pass


# ============================================================================
# Example 4: Custom Composition (Pick and Choose)
# ============================================================================

class CustomAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin):
    """Custom agent with only graph and prediction (no statistics, no LLM)."""
    
    def __init__(self, variables=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.init_graph_feature(variables=variables, edges=edges)
        self.init_prediction_feature(use_nonlinear=False)  # Linear only
    
    def _get_domain_schema(self):
        return None  # No LLM schema needed
    
    def _build_domain_prompt(self, task: str) -> str:
        return ""  # Not used without LLM
    
    def _domain_specific_setup(self):
        pass
    
    def run(self, initial_state: Dict[str, float], interventions: Optional[Dict[str, float]] = None):
        """Run predictions without LLM."""
        return self.predict(initial_state, interventions or {})


# ============================================================================
# Example 5: Using Module Registry (Programmatic Composition)
# ============================================================================

def create_agent_with_features(features: List[str], **configs):
    """
    Create an agent with specified features using the registry.
    
    Args:
        features: List of feature names ('graph', 'prediction', 'statistics', 'llm')
        **configs: Configuration for each feature
        
    Returns:
        Agent class
    """
    # Define a base agent
    class DynamicAgent(BaseSpecializedAgent):
        def _get_domain_schema(self):
            return create_default_schema(
                function_name="dynamic_analysis",
                description="Dynamic analysis",
                properties={"result": {"type": "string"}}
            )
        
        def _build_domain_prompt(self, task: str) -> str:
            return f"Task: {task}"
        
        def _domain_specific_setup(self):
            pass
    
    # Compose with features
    return compose_agent(
        DynamicAgent,
        features=features,
        feature_configs=configs
    )


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Minimal agent
    minimal = MinimalAgent(agent_name="minimal")
    # result = minimal.analyze_with_llm("Analyze this problem")
    
    # Example 2: Graph + LLM
    graph_llm = GraphLLMAgent(
        agent_name="graph-llm",
        variables=["A", "B", "C"],
        edges=[("A", "B"), ("B", "C")]
    )
    # result = graph_llm.analyze_with_llm("Analyze the graph")
    
    # Example 3: Full featured
    full = FullFeaturedAgent(
        agent_name="full",
        variables=["X", "Y", "Z"],
        edges=[("X", "Y"), ("Y", "Z")]
    )
    # predictions = full.predict({"X": 1.0}, {"X": 1.5})
    
    # Example 4: Custom composition
    custom = CustomAgent(
        variables=["A", "B"],
        edges=[("A", "B")]
    )
    # predictions = custom.run({"A": 1.0}, {"A": 2.0})
    
    # Example 5: Programmatic composition
    AgentClass = create_agent_with_features(
        features=['graph', 'prediction'],
        graph={'graph_type': 'causal'},
        prediction={'use_nonlinear': True}
    )
    # agent = AgentClass(agent_name="dynamic")
    
    print("All examples created successfully!")

