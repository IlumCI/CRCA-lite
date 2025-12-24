"""
Module Registry for Drag-and-Drop Feature Composition

Provides a registry system and mixin patterns to make modules truly composable
and "drag-and-drop" friendly for creating specialized agents.
"""

from typing import Dict, Any, List, Optional, Type, Callable
import logging
from abc import ABC

logger = logging.getLogger(__name__)


class FeatureMixin(ABC):
    """
    Base mixin class for feature modules.
    
    Features can be mixed into agents to add capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the feature mixin."""
        super().__init__(*args, **kwargs)
        self._feature_enabled = True
    
    def enable_feature(self) -> None:
        """Enable this feature."""
        self._feature_enabled = True
    
    def disable_feature(self) -> None:
        """Disable this feature."""
        self._feature_enabled = False
    
    def is_feature_enabled(self) -> bool:
        """Check if feature is enabled."""
        return getattr(self, '_feature_enabled', True)


class ModuleRegistry:
    """
    Registry for managing and composing feature modules.
    
    Allows "drag-and-drop" style composition of features.
    """
    
    _modules: Dict[str, Type] = {}
    _dependencies: Dict[str, List[str]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        module_class: Type,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a module for use in agents.
        
        Args:
            name: Module name (e.g., 'graph', 'prediction', 'statistics')
            module_class: Module class
            dependencies: List of required module names
        """
        cls._modules[name] = module_class
        cls._dependencies[name] = dependencies or []
    
    @classmethod
    def get_module(cls, name: str) -> Optional[Type]:
        """
        Get a registered module class.
        
        Args:
            name: Module name
            
        Returns:
            Module class or None if not found
        """
        return cls._modules.get(name)
    
    @classmethod
    def get_dependencies(cls, name: str) -> List[str]:
        """
        Get dependencies for a module.
        
        Args:
            name: Module name
            
        Returns:
            List of dependency names
        """
        return cls._dependencies.get(name, [])
    
    @classmethod
    def list_modules(cls) -> List[str]:
        """
        List all registered modules.
        
        Returns:
            List of module names
        """
        return list(cls._modules.keys())
    
    @classmethod
    def check_dependencies(cls, module_names: List[str]) -> Dict[str, List[str]]:
        """
        Check if all dependencies are satisfied.
        
        Args:
            module_names: List of module names to check
            
        Returns:
            Dictionary mapping module names to missing dependencies
        """
        missing = {}
        for module_name in module_names:
            deps = cls.get_dependencies(module_name)
            missing_deps = [d for d in deps if d not in module_names]
            if missing_deps:
                missing[module_name] = missing_deps
        return missing


class FeatureComposer:
    """
    Composer for building agents with selected features.
    
    Provides a fluent interface for "drag-and-drop" feature composition.
    """
    
    def __init__(self, base_agent_class: Type):
        """
        Initialize the feature composer.
        
        Args:
            base_agent_class: Base agent class to compose features into
        """
        self.base_agent_class = base_agent_class
        self.selected_features: List[str] = []
        self.feature_configs: Dict[str, Dict[str, Any]] = {}
    
    def with_feature(
        self,
        feature_name: str,
        **config
    ) -> 'FeatureComposer':
        """
        Add a feature to the agent.
        
        Args:
            feature_name: Name of the feature module
            **config: Configuration for the feature
            
        Returns:
            Self for method chaining
        """
        if feature_name not in self.selected_features:
            self.selected_features.append(feature_name)
        self.feature_configs[feature_name] = config
        return self
    
    def without_feature(self, feature_name: str) -> 'FeatureComposer':
        """
        Remove a feature from the agent.
        
        Args:
            feature_name: Name of the feature to remove
            
        Returns:
            Self for method chaining
        """
        if feature_name in self.selected_features:
            self.selected_features.remove(feature_name)
        if feature_name in self.feature_configs:
            del self.feature_configs[feature_name]
        return self
    
    def build(self) -> Type:
        """
        Build the agent class with selected features.
        
        Returns:
            Agent class with features composed
        """
        # Check dependencies
        missing = ModuleRegistry.check_dependencies(self.selected_features)
        if missing:
            raise ValueError(f"Missing dependencies: {missing}")
        
        # Create dynamic class with features
        class_name = f"{self.base_agent_class.__name__}WithFeatures"
        bases = (self.base_agent_class,)
        
        # Add feature initialization
        def __init__(self, *args, **kwargs):
            # Initialize base
            self.base_agent_class.__init__(self, *args, **kwargs)
            
            # Initialize features
            for feature_name in self.selected_features:
                feature_class = ModuleRegistry.get_module(feature_name)
                if feature_class:
                    config = self.feature_configs.get(feature_name, {})
                    # Auto-inject dependencies
                    if hasattr(feature_class, '_auto_inject_dependencies'):
                        deps = feature_class._auto_inject_dependencies()
                        for dep_name in deps:
                            if hasattr(self, dep_name):
                                config[dep_name] = getattr(self, dep_name)
                    
                    feature_instance = feature_class(**config)
                    setattr(self, feature_name, feature_instance)
        
        # Create class
        agent_class = type(class_name, bases, {
            '__init__': __init__,
            '_selected_features': self.selected_features,
        })
        
        return agent_class


def compose_agent(
    base_agent_class: Type,
    features: List[str],
    feature_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> Type:
    """
    Convenience function to compose an agent with features.
    
    Args:
        base_agent_class: Base agent class
        features: List of feature names to add
        feature_configs: Optional configuration for each feature
        
    Returns:
        Composed agent class
        
    Example:
        >>> MyAgent = compose_agent(
        ...     BaseSpecializedAgent,
        ...     features=['graph', 'prediction'],
        ...     feature_configs={
        ...         'graph': {'graph_type': 'causal'},
        ...         'prediction': {'use_nonlinear': True}
        ...     }
        ... )
    """
    composer = FeatureComposer(base_agent_class)
    for feature in features:
        config = feature_configs.get(feature, {}) if feature_configs else {}
        composer.with_feature(feature, **config)
    return composer.build()


# Register built-in modules
def register_builtin_modules():
    """Register built-in modules with the registry."""
    from .graph_management import GraphManager
    from .prediction_framework import PredictionFramework
    from .statistical_methods import StatisticalMethods
    from .llm_integration import LLMIntegration
    
    ModuleRegistry.register('graph', GraphManager)
    ModuleRegistry.register('prediction', PredictionFramework, dependencies=['graph'])
    ModuleRegistry.register('statistics', StatisticalMethods, dependencies=['graph', 'prediction'])
    ModuleRegistry.register('llm', LLMIntegration)


# Auto-register on import
try:
    register_builtin_modules()
except ImportError:
    # Modules not available yet, will be registered when imported
    pass

