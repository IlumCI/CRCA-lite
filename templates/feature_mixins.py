"""
Feature Mixins for Drag-and-Drop Composition

Provides mixin classes that can be easily mixed into agents to add capabilities.
Each mixin is self-contained and can be used independently.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .base_specialized_agent import BaseSpecializedAgent
    from .graph_management import GraphManager
    from .prediction_framework import PredictionFramework
    from .statistical_methods import StatisticalMethods
    from .llm_integration import LLMIntegration

logger = logging.getLogger(__name__)


class GraphFeatureMixin:
    """
    Mixin to add graph management capabilities.
    
    Usage:
        class MyAgent(BaseSpecializedAgent, GraphFeatureMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_graph_feature(graph_type='causal')
    """
    
    def init_graph_feature(
        self,
        graph_type: str = "causal",
        variables: Optional[List[str]] = None,
        edges: Optional[List[tuple]] = None,
    ) -> None:
        """
        Initialize graph feature.
        
        Args:
            graph_type: Type of graph
            variables: Optional list of variables
            edges: Optional list of (source, target) tuples
        """
        from .graph_management import GraphManager
        
        self.graph_manager = GraphManager(graph_type=graph_type)
        
        if variables:
            self.graph_manager.add_nodes_from(variables)
        if edges:
            self.graph_manager.add_edges_from(edges)
    
    # Convenience methods
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        if hasattr(self, 'graph_manager'):
            self.graph_manager.ensure_node_exists(node)
    
    def add_edge(self, source: str, target: str, **metadata) -> None:
        """Add an edge to the graph."""
        if hasattr(self, 'graph_manager'):
            self.graph_manager.add_relationship(source, target, **metadata)
    
    def get_nodes(self) -> List[str]:
        """Get all nodes."""
        if hasattr(self, 'graph_manager'):
            return self.graph_manager.get_nodes()
        return []
    
    def get_edges(self) -> List[tuple]:
        """Get all edges."""
        if hasattr(self, 'graph_manager'):
            return self.graph_manager.get_edges()
        return []


class PredictionFeatureMixin:
    """
    Mixin to add prediction capabilities.
    
    Usage:
        class MyAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_graph_feature()
                self.init_prediction_feature()
    """
    
    def init_prediction_feature(
        self,
        use_nonlinear: bool = True,
        cache_enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize prediction feature.
        
        Args:
            use_nonlinear: Whether to use nonlinear predictions
            cache_enabled: Whether to enable caching
            **kwargs: Additional prediction framework config
        """
        from .prediction_framework import PredictionFramework
        
        if not hasattr(self, 'graph_manager'):
            raise ValueError("GraphFeatureMixin required before PredictionFeatureMixin")
        
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager,
            use_nonlinear=use_nonlinear,
            cache_enabled=cache_enabled,
            **kwargs
        )
    
    def predict(self, state: Dict[str, float], interventions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Predict outcomes."""
        if hasattr(self, 'prediction_framework'):
            return self.prediction_framework.predict_outcomes(state, interventions or {})
        raise AttributeError("Prediction feature not initialized. Call init_prediction_feature() first.")


class StatisticsFeatureMixin:
    """
    Mixin to add statistical analysis capabilities.
    
    Usage:
        class MyAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin, StatisticsFeatureMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_graph_feature()
                self.init_prediction_feature()
                self.init_statistics_feature()
    """
    
    def init_statistics_feature(
        self,
        seed: int = 42,
        bootstrap_workers: int = 0,
        **kwargs
    ) -> None:
        """
        Initialize statistics feature.
        
        Args:
            seed: Random seed
            bootstrap_workers: Number of bootstrap workers
            **kwargs: Additional statistics config
        """
        from .statistical_methods import StatisticalMethods
        
        if not hasattr(self, 'graph_manager'):
            raise ValueError("GraphFeatureMixin required before StatisticsFeatureMixin")
        if not hasattr(self, 'prediction_framework'):
            raise ValueError("PredictionFeatureMixin required before StatisticsFeatureMixin")
        
        self.statistical_methods = StatisticalMethods(
            graph_manager=self.graph_manager,
            prediction_framework=self.prediction_framework,
            standardization_stats=getattr(self.prediction_framework, 'standardization_stats', {}),
            seed=seed,
            bootstrap_workers=bootstrap_workers,
            **kwargs
        )
    
    def fit_from_data(self, df: Any, variables: List[str], **kwargs) -> None:
        """Fit model from data."""
        if hasattr(self, 'statistical_methods'):
            self.statistical_methods.fit_from_dataframe(df, variables, **kwargs)
        else:
            raise AttributeError("Statistics feature not initialized. Call init_statistics_feature() first.")


class LLMFeatureMixin:
    """
    Mixin to add LLM integration capabilities.
    
    Usage:
        class MyAgent(BaseSpecializedAgent, LLMFeatureMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_llm_feature()
    """
    
    def init_llm_feature(
        self,
        max_loops: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize LLM feature.
        
        Args:
            max_loops: Maximum reasoning loops
            **kwargs: Additional LLM config
        """
        from .llm_integration import LLMIntegration
        
        max_loops = max_loops or getattr(self, 'domain_max_loops', 3)
        
        self.llm_integration = LLMIntegration(
            agent=self,
            max_loops=max_loops,
            **kwargs
        )
    
    def analyze_with_llm(self, task: str, **kwargs) -> Dict[str, Any]:
        """Run LLM-based analysis."""
        if hasattr(self, 'llm_integration'):
            return self.llm_integration.run_llm_domain_analysis(
                task,
                build_prompt_fn=getattr(self, '_build_domain_prompt', None),
                **kwargs
            )
        raise AttributeError("LLM feature not initialized. Call init_llm_feature() first.")


# Convenience: All-in-one mixin
class FullFeatureMixin(GraphFeatureMixin, PredictionFeatureMixin, StatisticsFeatureMixin, LLMFeatureMixin):
    """
    Mixin that includes all features.
    
    Usage:
        class MyAgent(BaseSpecializedAgent, FullFeatureMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_all_features()
    """
    
    def init_all_features(
        self,
        graph_type: str = "causal",
        use_nonlinear: bool = True,
        variables: Optional[List[str]] = None,
        edges: Optional[List[tuple]] = None,
        **kwargs
    ) -> None:
        """
        Initialize all features at once.
        
        Args:
            graph_type: Type of graph
            use_nonlinear: Whether to use nonlinear predictions
            variables: Optional list of variables
            edges: Optional list of edges
            **kwargs: Additional config for individual features
        """
        self.init_graph_feature(graph_type=graph_type, variables=variables, edges=edges)
        self.init_prediction_feature(use_nonlinear=use_nonlinear, **kwargs.get('prediction', {}))
        self.init_statistics_feature(**kwargs.get('statistics', {}))
        self.init_llm_feature(**kwargs.get('llm', {}))

