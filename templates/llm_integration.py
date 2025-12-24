"""
LLM integration module for specialized agents.

Provides patterns for schema definition, prompt building, memory management,
and multi-loop reasoning with LLMs.
"""

from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    LLM integration patterns for specialized agents.
    
    Provides methods for:
    - Schema definition
    - Prompt building
    - Memory management
    - Multi-loop reasoning
    """

    def __init__(
        self,
        agent: Any,  # BaseSpecializedAgent instance
        max_loops: int = 3,
    ):
        """
        Initialize LLM integration.
        
        Args:
            agent: BaseSpecializedAgent instance
            max_loops: Maximum number of reasoning loops
        """
        self.agent = agent
        self.max_loops = max_loops
        self.domain_memory: List[Dict[str, Any]] = []

    def build_domain_prompt(self, task: str, **context) -> str:
        """
        Build a domain-specific prompt for the given task.
        
        This is a template method that should be overridden by subclasses
        or customized per domain.
        
        Args:
            task: The task description
            **context: Additional context variables
            
        Returns:
            Formatted prompt string
        """
        # Default implementation - should be overridden
        return f"Task: {task}\n\nPlease analyze this task and provide a comprehensive response."

    def build_memory_context(self, last_n: int = 2) -> str:
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

    def synthesize_analysis(self, task: str) -> str:
        """
        Synthesize a final analysis from the reasoning steps.
        
        Args:
            task: The original task
            
        Returns:
            Synthesized analysis string
        """
        synthesis_prompt = f"Based on the analysis steps performed, synthesize a concise report for: {task}"
        return self.agent.step(synthesis_prompt)

    def run_llm_domain_analysis(
        self,
        task: str,
        build_prompt_fn: Optional[Callable[[str], str]] = None,
        post_process_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multi-loop LLM-based domain analysis.
        
        Args:
            task: The task to analyze
            build_prompt_fn: Optional function to build prompts (defaults to build_domain_prompt)
            post_process_fn: Optional function to post-process results
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        self.domain_memory = []

        # Use provided prompt builder or default
        if build_prompt_fn:
            domain_prompt = build_prompt_fn(task)
        else:
            domain_prompt = self.build_domain_prompt(task)

        max_loops = self.max_loops if isinstance(self.max_loops, int) else 3
        for i in range(max_loops):
            step_result = self.agent.step(domain_prompt)
            self.domain_memory.append({
                'step': i + 1,
                'analysis': step_result,
                'timestamp': i
            })

            if i < max_loops - 1:
                memory_context = self.build_memory_context()
                domain_prompt = f"{domain_prompt}\n\nPrevious Analysis:\n{memory_context}"

        final_analysis = self.synthesize_analysis(task)

        result = {
            'task': task,
            'analysis': final_analysis,
            'analysis_steps': self.domain_memory,
            **kwargs
        }

        # Apply post-processing if provided
        if post_process_fn:
            result = post_process_fn(result)

        return result

    def clear_memory(self) -> None:
        """Clear the domain memory."""
        self.domain_memory = []

    def get_memory(self) -> List[Dict[str, Any]]:
        """
        Get the current domain memory.
        
        Returns:
            List of memory entries
        """
        return self.domain_memory.copy()

    def add_to_memory(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to the domain memory.
        
        Args:
            entry: Memory entry dictionary
        """
        self.domain_memory.append(entry)


def create_default_schema(
    function_name: str,
    description: str,
    properties: Dict[str, Dict[str, Any]],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a default function schema for LLM tools.
    
    Args:
        function_name: Name of the function
        description: Description of the function
        properties: Dictionary of parameter properties
        required: List of required parameters
        
    Returns:
        Function schema dictionary
    """
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or list(properties.keys())
            }
        }
    }

