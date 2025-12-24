"""
Base template class for specialized agents.

Provides common patterns for initialization, schema handling, memory management,
and LLM integration that can be extended by domain-specific agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import threading
import logging
from swarms.structs.agent import Agent

logger = logging.getLogger(__name__)


class BaseSpecializedAgent(Agent, ABC):
    """
    Base template class for creating specialized agents.
    
    This class provides common patterns for:
    - Schema-based tool definition
    - Agent initialization with domain-specific setup
    - Memory management for multi-loop reasoning
    - Caching infrastructure
    - Async wrapper utilities
    
    Subclasses should implement:
    - _get_domain_schema(): Return the tool schema for this domain
    - _build_domain_prompt(task: str): Build domain-specific prompt
    - _domain_specific_setup(): Initialize domain-specific attributes
    """

    def __init__(
        self,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "specialized-agent",
        agent_description: str = "Specialized Agent",
        description: Optional[str] = None,
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the specialized agent.
        
        Args:
            max_loops: Maximum number of reasoning loops
            agent_name: Unique identifier for the agent
            agent_description: Human-readable description
            description: Backwards-compatible alias for agent_description
            model_name: LLM model to use
            system_prompt: System-level instructions
            global_system_prompt: Global system prompt
            secondary_system_prompt: Secondary system prompt
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to base Agent
        """
        # Get domain-specific schema
        domain_schema = self._get_domain_schema()
        
        # Backwards-compatible alias for description
        agent_description = description or agent_description

        # Prepare agent kwargs
        agent_kwargs = {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "model_name": model_name,
            "max_loops": 1,  # Individual LLM calls use 1 loop, we handle multi-loop reasoning
            "tools_list_dictionary": [domain_schema] if domain_schema else [],
            "output_type": "final",
            **kwargs,
        }
        
        # Add optional prompts
        if system_prompt is not None:
            agent_kwargs["system_prompt"] = system_prompt
        if global_system_prompt is not None:
            agent_kwargs["global_system_prompt"] = global_system_prompt
        if secondary_system_prompt is not None:
            agent_kwargs["secondary_system_prompt"] = secondary_system_prompt
        
        # Initialize base Agent
        super().__init__(**agent_kwargs)
        
        # Store max loops for multi-loop reasoning
        self.domain_max_loops = max_loops
        self.seed = seed if seed is not None else 42
        
        # Initialize memory for multi-loop reasoning
        self.domain_memory: List[Dict[str, Any]] = []
        
        # Initialize caching infrastructure
        self._prediction_cache: Dict[Any, Any] = {}
        self._prediction_cache_order: List[Any] = []
        self._prediction_cache_max: int = 1000
        self._cache_enabled: bool = True
        self._prediction_cache_lock = threading.Lock()
        
        # Call domain-specific setup
        self._domain_specific_setup()

    @abstractmethod
    def _get_domain_schema(self) -> Optional[Dict[str, Any]]:
        """
        Return the tool schema for this domain.
        
        Returns:
            Dict containing the function schema, or None if no schema needed
        """
        pass

    @abstractmethod
    def _build_domain_prompt(self, task: str) -> str:
        """
        Build a domain-specific prompt for the given task.
        
        Args:
            task: The task description
            
        Returns:
            Formatted prompt string
        """
        pass

    def _domain_specific_setup(self) -> None:
        """
        Initialize domain-specific attributes.
        
        Override this method in subclasses to set up domain-specific
        data structures, graphs, models, etc.
        """
        pass

    def step(self, task: str) -> str:
        """
        Execute a single step of reasoning.
        
        Args:
            task: The task to execute
            
        Returns:
            Response string from the agent
        """
        response = super().run(task)
        return response

    def _build_memory_context(self, last_n: int = 2) -> str:
        """
        Build context from recent memory steps.
        
        Args:
            last_n: Number of recent steps to include
            
        Returns:
            Formatted memory context string
        """
        context_parts = []
        for step in self.domain_memory[-last_n:]:
            context_parts.append(f"Step {step.get('step', '?')}: {step.get('analysis', '')}")
        return "\n".join(context_parts)

    def _synthesize_analysis(self, task: str) -> str:
        """
        Synthesize a final analysis from the reasoning steps.
        
        Args:
            task: The original task
            
        Returns:
            Synthesized analysis string
        """
        synthesis_prompt = f"Based on the analysis steps performed, synthesize a concise report for: {task}"
        return self.step(synthesis_prompt)

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        with self._prediction_cache_lock:
            self._prediction_cache.clear()
            self._prediction_cache_order.clear()

    def enable_cache(self, flag: bool) -> None:
        """
        Enable or disable caching.
        
        Args:
            flag: True to enable, False to disable
        """
        with self._prediction_cache_lock:
            self._cache_enabled = bool(flag)

