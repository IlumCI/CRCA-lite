"""
Template framework for creating specialized agents.

This package provides a base template class and modular components for building
domain-specific agents with LLM integration, graph management, prediction frameworks,
and statistical methods.

Features can be used as "drag-and-drop" components via mixins or the module registry.
"""

from typing import Optional

try:
    from .base_specialized_agent import BaseSpecializedAgent
    from .feature_mixins import (
        GraphFeatureMixin,
        PredictionFeatureMixin,
        StatisticsFeatureMixin,
        LLMFeatureMixin,
        FullFeatureMixin
    )
    from .module_registry import ModuleRegistry, compose_agent, FeatureComposer
    __all__ = [
        "BaseSpecializedAgent",
        "GraphFeatureMixin",
        "PredictionFeatureMixin",
        "StatisticsFeatureMixin",
        "LLMFeatureMixin",
        "FullFeatureMixin",
        "ModuleRegistry",
        "compose_agent",
        "FeatureComposer",
    ]
except ImportError:
    __all__ = []

__version__ = "0.2.0"

