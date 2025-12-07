"""
CR-CA Lite: A lightweight Causal Reasoning with Counterfactual Analysis Agent.

This is a lightweight Agent implementation of the ASTT/CR-CA framework for the swarms
framework. It provides core causal reasoning capabilities with LLM integration, supporting
both sophisticated LLM-based causal analysis and deterministic causal simulation.

Core capabilities:
- LLM-based causal analysis: Multi-loop reasoning with structured output (like full CRCAAgent)
- Evolution operator E(x): Deterministic causal state evolution
- Counterfactual scenario generation: Systematic "what-if" analysis
- Causal chain identification: Path finding in causal graphs
- Basic causal graph operations: DAG management and traversal

This Agent inherits from swarms.structs.agent.Agent, making it a full Agent in the
swarms framework with LLM integration. It supports dual-mode operation:
- LLM mode: When task is a string, performs sophisticated causal reasoning using LLM
- Deterministic mode: When initial_state is a dict, performs pure causal simulation

Dependencies: numpy + swarms.structs.agent (typing, dataclasses, enum are stdlib)
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
from swarms.structs.agent import Agent


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class CausalNode:
    """Represents a node in the causal graph"""
    name: str
    value: Optional[float] = None
    confidence: float = 1.0
    node_type: str = "variable"


@dataclass
class CausalEdge:
    """Represents an edge in the causal graph"""
    source: str
    target: str
    strength: float = 1.0
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    confidence: float = 1.0


@dataclass
class CounterfactualScenario:
    """Represents a counterfactual scenario"""
    name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float = 1.0
    reasoning: str = ""


class CRCAAgent(Agent):
    """
    CR-CA Lite: Lightweight Causal Reasoning with Counterfactual Analysis Agent.
    
    A deterministic Agent in the swarms framework that performs causal reasoning
    and counterfactual analysis without requiring LLM dependencies or external APIs.
    This Agent inherits from swarms.structs.agent.Agent and implements the core
    ASTT/CR-CA primitives for causal simulation.
    
    Core components:
    - Evolution operator: E(x) = _predict_outcomes() - deterministic state evolution
    - Counterfactual generation: generate_counterfactual_scenarios() - systematic intervention analysis
    - Causal chain identification: identify_causal_chain() - path finding in causal DAGs
    - State mapping: _standardize_state() / _destandardize_value() - z-space transformations
    
    Agent Interface:
    - Inherits from swarms.structs.agent.Agent for full framework compatibility
    - Overrides `run()` method for causal reasoning tasks
    - Compatible with swarms workflows, orchestrators, and agent composition patterns
    
    Args:
        variables: Optional list of variable names to initialize in the causal graph
        causal_edges: Optional list of (source, target) tuples for initial causal relationships
        max_loops: Maximum evolution steps (int) or "auto" for automatic determination
        **kwargs: Additional arguments passed to Agent base class
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "cr-ca-lite-agent",
        agent_description: str = "Lightweight Causal Reasoning with Counterfactual Analysis Agent",
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize CR-CA Lite Agent.
        
        Args:
            variables: Optional list of variable names to add to graph
            causal_edges: Optional list of (source, target) tuples for initial edges
            max_loops: Maximum reasoning loops for LLM-based causal analysis (default: 3)
            agent_name: Name of the agent
            agent_description: Description of the agent
            model_name: LLM model to use for causal reasoning
            system_prompt: Optional custom system prompt
            global_system_prompt: Optional global system prompt
            secondary_system_prompt: Optional secondary system prompt
            **kwargs: Additional arguments passed to Agent base class
        """
        # Get CR-CA schema for function calling (static method, callable before super().__init__)
        cr_ca_schema = CRCAAgent._get_cr_ca_schema()
        
        # Prepare agent kwargs with CR-CA schema
        agent_kwargs = {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "model_name": model_name,
            "max_loops": 1,  # Individual LLM calls use 1 loop, we handle multi-loop reasoning
            "tools_list_dictionary": [cr_ca_schema],
            "output_type": "final",
            **kwargs,
        }
        
        # Add optional system prompts if provided
        if system_prompt is not None:
            agent_kwargs["system_prompt"] = system_prompt
        if global_system_prompt is not None:
            agent_kwargs["global_system_prompt"] = global_system_prompt
        if secondary_system_prompt is not None:
            agent_kwargs["secondary_system_prompt"] = secondary_system_prompt
        
        # Initialize Agent base class with LLM integration
        super().__init__(**agent_kwargs)
        
        # Store max_loops for multi-step causal reasoning
        self.causal_max_loops = max_loops
        # Pure Python graph representation: {node: {child: strength}}
        self.causal_graph: Dict[str, Dict[str, float]] = {}
        self.causal_graph_reverse: Dict[str, List[str]] = {}  # For fast parent lookup
        
        # Standardization statistics: {'var': {'mean': m, 'std': s}}
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        
        # Initialize graph
        if variables:
            for var in variables:
                self._ensure_node_exists(var)

        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

        # Memory for storing causal analysis history (LLM-based)
        self.causal_memory: List[Dict[str, Any]] = []

    @staticmethod
    def _get_cr_ca_schema() -> Dict[str, Any]:
        """Get the CR-CA agent schema for structured reasoning."""
        return {
            "type": "function",
            "function": {
                "name": "generate_causal_analysis",
                "description": "Generates structured causal reasoning and counterfactual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
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
                    "required": [
                        "causal_analysis",
                        "intervention_planning", 
                        "counterfactual_scenarios",
                        "causal_strength_assessment",
                        "optimal_solution"
                    ]
                }
            }
        }

    def step(self, task: str) -> str:
        """Execute a single step of causal reasoning using LLM."""
        response = super().run(task)
        return response

    def _build_causal_prompt(self, task: str) -> str:
        """Build the causal analysis prompt."""
        return f"""
        You are a Causal Reasoning with Counterfactual Analysis (CR-CA) agent. 
        Analyze the following problem using sophisticated causal reasoning:
        
        Problem: {task}
        
        Your analysis should include:
        1. Causal Analysis: Identify cause-and-effect relationships
        2. Intervention Planning: Plan interventions to test causal hypotheses  
        3. Counterfactual Scenarios: Explore multiple "what-if" scenarios
        4. Causal Strength Assessment: Evaluate relationship strengths and confounders
        5. Optimal Solution: Recommend the best approach based on causal analysis
        
        Current causal graph has {len(self.causal_graph)} variables and {sum(len(children) for children in self.causal_graph.values())} relationships.
        """

    def _build_memory_context(self) -> str:
        """Build memory context from previous analysis steps."""
        context_parts = []
        for step in self.causal_memory[-2:]:  # Last 2 steps
            context_parts.append(f"Step {step['step']}: {step['analysis']}")
        return "\n".join(context_parts)

    def _synthesize_causal_analysis(self, task: str) -> str:
        """Synthesize the final causal analysis from all steps."""
        synthesis_prompt = f"""
        Based on the causal analysis steps performed, synthesize a comprehensive 
        causal reasoning report for: {task}
        
        Include:
        - Key causal relationships identified
        - Recommended interventions
        - Counterfactual scenarios explored
        - Optimal solution with causal justification
        - Confidence levels and limitations
        """
        
        return self.step(synthesis_prompt)

    def _ensure_node_exists(self, node: str) -> None:
        """Ensure node present in graph structures."""
        if node not in self.causal_graph:
            self.causal_graph[node] = {}
        if node not in self.causal_graph_reverse:
            self.causal_graph_reverse[node] = []

    def add_causal_relationship(
        self, 
        source: str, 
        target: str, 
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        """
        Add a causal edge to the graph.
        
        Args:
            source: Source variable name
            target: Target variable name
            strength: Causal effect strength (default: 1.0)
            relation_type: Type of causal relation (default: DIRECT)
            confidence: Confidence in the relationship (default: 1.0)
        """
        # Ensure nodes exist
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)

        # Add or update edge: source -> target with strength
        self.causal_graph[source][target] = float(strength)

        # Update reverse mapping for parent lookup (avoid duplicates)
        if source not in self.causal_graph_reverse[target]:
            self.causal_graph_reverse[target].append(source)
    
    def _get_parents(self, node: str) -> List[str]:
        """
        Get parent nodes (predecessors) of a node.
        
        Args:
            node: Node name
        
        Returns:
            List of parent node names
        """
        return self.causal_graph_reverse.get(node, [])
    
    def _get_children(self, node: str) -> List[str]:
        """
        Get child nodes (successors) of a node.
        
        Args:
            node: Node name
        
        Returns:
            List of child node names
        """
        return list(self.causal_graph.get(node, {}).keys())
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm (pure Python).
        
        Returns:
            List of nodes in topological order
        """
        # Compute in-degrees
        in_degree: Dict[str, int] = {node: 0 for node in self.causal_graph.keys()}
        for node in self.causal_graph:
            for child in self._get_children(node):
                in_degree[child] = in_degree.get(child, 0) + 1
        
        # Initialize queue with nodes having no incoming edges
        queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
        result: List[str] = []
        
        # Process nodes
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Reduce in-degree of children
            for child in self._get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def identify_causal_chain(self, start: str, end: str) -> List[str]:
        """
        Find shortest causal path from start to end using BFS (pure Python).
        
        Implements core causal chain identification (Ax2, Ax6).
        
        Args:
            start: Starting variable
            end: Target variable
        
        Returns:
            List of variables forming the causal chain, or empty list if no path exists
        """
        if start not in self.causal_graph or end not in self.causal_graph:
            return []
        
        if start == end:
            return [start]
        
        # BFS to find shortest path
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: set = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            # Check all children
            for child in self._get_children(current):
                if child == end:
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []  # No path found
    
    # detect_confounders removed in Lite version (advanced inference)
    
    def _has_path(self, start: str, end: str) -> bool:
        """
        Check if a path exists from start to end using DFS.
        
        Args:
            start: Starting node
            end: Target node
        
        Returns:
            True if path exists, False otherwise
        """
        if start == end:
            return True
        
        stack = [start]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for child in self._get_children(current):
                if child == end:
                    return True
                if child not in visited:
                    stack.append(child)
        
        return False
    
    # identify_adjustment_set removed in Lite version (advanced inference)
    
    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Standardize state values to z-scores.
        
        Args:
            state: Dictionary of variable values
        
        Returns:
            Dictionary of standardized (z-score) values
        """
        z: Dict[str, float] = {}
        for k, v in state.items():
            s = self.standardization_stats.get(k)
            if s and s.get("std", 0.0) > 0:
                z[k] = (v - s["mean"]) / s["std"]
            else:
                z[k] = v
        return z
    
    def _destandardize_value(self, var: str, z_value: float) -> float:
        """
        Convert z-score back to original scale.
        
        Args:
            var: Variable name
            z_value: Standardized (z-score) value
        
        Returns:
            Original scale value
        """
        s = self.standardization_stats.get(var)
        if s and s.get("std", 0.0) > 0:
            return z_value * s["std"] + s["mean"]
        return z_value
    
    def _predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evolution operator E(x): Predict outcomes given state and interventions.
        
        This is the core CR-CA evolution operator implementing:
        x_{t+1} = E(x_t)
        
        Mathematical foundation:
        - Linear structural causal model: y = Σᵢ βᵢ·xᵢ + ε\n+        - NOTE: This implementation is linear. To model nonlinear dynamics override\n+          `_predict_outcomes` in a subclass with a custom evolution operator.\n*** End Patch
        - Propagates effects through causal graph in topological order
        - Standardizes inputs, computes in z-space, de-standardizes outputs
        
        Args:
            factual_state: Current world state (baseline)
            interventions: Interventions to apply (do-operator)
        
        Returns:
            Dictionary of predicted variable values
        """
        # Merge factual state with interventions
        raw = factual_state.copy()
        raw.update(interventions)
        
        # Standardize to z-scores
        z_state = self._standardize_state(raw)
        z_pred = dict(z_state)
        
        # Process nodes in topological order
        for node in self._topological_sort():
            # If node is intervened on, keep its value
            if node in interventions:
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue
            
            # Get parents
            parents = self._get_parents(node)
            if not parents:
                continue
            
            # Compute linear combination: Σᵢ βᵢ·z_xi
            s = 0.0
            for p in parents:
                pz = z_pred.get(p, z_state.get(p, 0.0))
                strength = self.causal_graph.get(p, {}).get(node, 0.0)
                s += pz * strength
            
            z_pred[node] = s
        
        # De-standardize results
        return {v: self._destandardize_value(v, z) for v, z in z_pred.items()}
    
    def _calculate_scenario_probability(
        self,
        factual_state: Dict[str, float], 
        interventions: Dict[str, float]
    ) -> float:
        """
        Calculate a heuristic probability of a counterfactual scenario.
        
        NOTE: This is a lightweight heuristic proximity measure (Mahalanobis-like)
        and NOT a full statistical estimator — it ignores covariance and should
        be treated as a relative plausibility score for Lite usage.
        
        Args:
            factual_state: Baseline state
            interventions: Intervention values
        
        Returns:
            Heuristic probability value between 0.05 and 0.98
        """
        z_sq = 0.0
        for var, new in interventions.items():
            s = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            mu, sd = s.get("mean", 0.0), s.get("std", 1.0) or 1.0
            old = factual_state.get(var, mu)
            dz = (new - mu) / sd - (old - mu) / sd
            z_sq += float(dz) * float(dz)
        
        p = 0.95 * float(np.exp(-0.5 * z_sq)) + 0.05
        return float(max(0.05, min(0.98, p)))
    
    def generate_counterfactual_scenarios(
        self,
        factual_state: Dict[str, float],
        target_variables: List[str],
        max_scenarios: int = 5
    ) -> List[CounterfactualScenario]:
        """
        Generate counterfactual scenarios for target variables.
        
        Implements Ax8 (Counterfactuals) - core CR-CA functionality.
        
        Args:
            factual_state: Current factual state
            target_variables: Variables to generate counterfactuals for
            max_scenarios: Maximum number of scenarios per variable
        
        Returns:
            List of CounterfactualScenario objects
        """
        # Ensure stats exist for variables in factual_state (fallback behavior)
        self.ensure_standardization_stats(factual_state)

        scenarios: List[CounterfactualScenario] = []
        z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        for i, tv in enumerate(target_variables[:max_scenarios]):
            stats = self.standardization_stats.get(tv, {"mean": 0.0, "std": 1.0})
            cur = factual_state.get(tv, stats.get("mean", 0.0))

            # If std is zero or missing, use absolute perturbations instead
            if not stats or stats.get("std", 0.0) <= 0:
                base = cur
                abs_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
                vals = [base + step for step in abs_steps]
            else:
                mean = stats["mean"]
                std = stats["std"]
                cz = (cur - mean) / std
                vals = [(cz + dz) * std + mean for dz in z_steps]

            for j, v in enumerate(vals):
                interventions = {tv: float(v)}
                scenarios.append(
                    CounterfactualScenario(
                        name=f"scenario_{i}_{j}",
                        interventions=interventions,
                        expected_outcomes=self._predict_outcomes(
                            factual_state, interventions
                        ),
                        probability=self._calculate_scenario_probability(
                            factual_state, interventions
                        ),
                        reasoning=f"Intervention on {tv} with value {v}",
                    )
                )

        return scenarios
    
    def analyze_causal_strength(self, source: str, target: str) -> Dict[str, float]:
        """
        Analyze the strength of causal relationship between two variables.
        
        Args:
            source: Source variable
            target: Target variable
        
        Returns:
            Dictionary with strength, confidence, path_length, relation_type
        """
        if source not in self.causal_graph or target not in self.causal_graph[source]:
            return {"strength": 0.0, "confidence": 0.0, "path_length": float('inf')}
        
        strength = self.causal_graph[source].get(target, 0.0)
        path = self.identify_causal_chain(source, target)
        path_length = len(path) - 1 if path else float('inf')
        
        return {
            "strength": float(strength),
            "confidence": 1.0,  # Simplified: assume full confidence
            "path_length": path_length,
            "relation_type": CausalRelationType.DIRECT.value
        }
    
    def set_standardization_stats(
        self,
        variable: str,
        mean: float,
        std: float
    ) -> None:
        """
        Set standardization statistics for a variable.
        
        Args:
            variable: Variable name
            mean: Mean value
            std: Standard deviation
        """
        self.standardization_stats[variable] = {"mean": mean, "std": std if std > 0 else 1.0}
    
    def ensure_standardization_stats(self, state: Dict[str, float]) -> None:
        """
        Ensure standardization stats exist for all variables in a given state.
        If stats are missing, create a sensible fallback (mean=observed, std=1.0).
        This prevents degenerate std=0 issues in Lite mode.
        """
        for var, val in state.items():
            if var not in self.standardization_stats:
                self.standardization_stats[var] = {"mean": float(val), "std": 1.0}
    
    def get_nodes(self) -> List[str]:
        """
        Get all nodes in the causal graph.
        
        Returns:
            List of node names
        """
        return list(self.causal_graph.keys())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Get all edges in the causal graph.
        
        Returns:
            List of (source, target) tuples
        """
        edges = []
        for source, targets in self.causal_graph.items():
            for target in targets.keys():
                edges.append((source, target))
        return edges
    
    def is_dag(self) -> bool:
        """
        Check if the causal graph is a DAG (no cycles).
        
        Uses DFS to detect cycles.
        
        Returns:
            True if DAG, False if cycles exist
        """
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            
            for child in self._get_children(node):
                if child not in visited:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        rec_stack = set()
        
        for node in self.causal_graph:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    return False
        
        return True
    
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Agent interface method: Run causal simulation and generate counterfactuals.
        
        Overrides Agent.run() to provide deterministic causal reasoning. Accepts either
        a task string (treated as JSON) or direct initial_state dict/JSON for compatibility
        with swarms Agent interface.
        
        Args:
            task: Task string (if provided, treated as JSON-encoded initial_state)
            initial_state: Initial world state as dict or JSON string (Agent-compatible input)
            target_variables: Variables to generate counterfactuals for (default: all nodes)
            max_steps: Number of evolution steps (default: 1) or "auto" for automatic
            **kwargs: Additional arguments for Agent compatibility
        
        Returns:
            Dictionary with:
            - evolved_state: Final state after causal evolution
            - counterfactual_scenarios: List of CounterfactualScenario objects
            - causal_graph_info: Graph structure metadata
            - initial_state: Original input state
            - steps: Number of evolution steps performed
        """
        # Mode 1: LLM-based causal analysis (like full CRCAAgent)
        if task is not None and isinstance(task, str) and initial_state is None and not task.strip().startswith('{'):
            return self._run_llm_causal_analysis(task, **kwargs)
        
        # Mode 2: Deterministic simulation (original Lite behavior)
        # Handle Agent interface: task parameter takes precedence if provided
        if task is not None and initial_state is None:
            initial_state = task
        
        # Accept either a dict initial_state or a JSON string (Agent interface compatibility)
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

        # Use all nodes as targets if not specified
        if target_variables is None:
            target_variables = list(self.causal_graph.keys())
        
        # Resolve "auto" sentinel for max_steps (accepts method arg or instance-level default)
        def _resolve_max_steps(value: Union[int, str]) -> int:
            if isinstance(value, str) and value == "auto":
                # Heuristic: one step per variable (at least 1)
                return max(1, len(self.causal_graph))
            try:
                return int(value)
            except Exception:
                return max(1, len(self.causal_graph))

        effective_steps = _resolve_max_steps(max_steps if max_steps != 1 or self.causal_max_loops == 1 else self.causal_max_loops)
        # If caller passed default 1 and instance set a different causal_max_loops, prefer instance value
        if max_steps == 1 and self.causal_max_loops != 1:
            effective_steps = _resolve_max_steps(self.causal_max_loops)

        # Evolve state
        current_state = initial_state.copy()
        for step in range(effective_steps):
            current_state = self._predict_outcomes(current_state, {})
        
        # Ensure standardization stats exist for the evolved state and generate counterfactuals from it
        self.ensure_standardization_stats(current_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            current_state,
            target_variables,
            max_scenarios=5
        )
        
        return {
            "initial_state": initial_state,
            "evolved_state": current_state,
            "counterfactual_scenarios": counterfactual_scenarios,
            "causal_graph_info": {
                "nodes": self.get_nodes(),
                "edges": self.get_edges(),
                "is_dag": self.is_dag()
            },
            "steps": effective_steps
        }

    def _run_llm_causal_analysis(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Run LLM-based causal analysis (like full CRCAAgent).
        
        Args:
            task: The problem or question to analyze causally
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing causal analysis results
        """
        # Reset memory
        self.causal_memory = []
        
        # Build causal analysis prompt
        causal_prompt = self._build_causal_prompt(task)
        
        # Run causal analysis with multiple loops
        max_loops = self.causal_max_loops if isinstance(self.causal_max_loops, int) else 3
        for i in range(max_loops):
            step_result = self.step(causal_prompt)
            self.causal_memory.append({
                'step': i + 1,
                'analysis': step_result,
                'timestamp': i
            })
            
            # Update prompt with previous analysis
            if i < max_loops - 1:
                memory_context = self._build_memory_context()
                causal_prompt = f"{causal_prompt}\n\nPrevious Analysis:\n{memory_context}"
        
        # Generate final causal analysis
        final_analysis = self._synthesize_causal_analysis(task)
        
        # Generate counterfactual scenarios using deterministic methods
        # Use a default state if available, or create one from graph nodes
        default_state = {var: 0.0 for var in self.get_nodes()}
        self.ensure_standardization_stats(default_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            default_state,
            self.get_nodes()[:5],  # Top 5 variables
            max_scenarios=5
        )
        
        return {
            'task': task,
            'causal_analysis': final_analysis,
            'counterfactual_scenarios': counterfactual_scenarios,
            'causal_graph_info': {
                'nodes': self.get_nodes(),
                'edges': self.get_edges(),
                'is_dag': self.is_dag()
            },
            'analysis_steps': self.causal_memory
        }


# Agent Interface: `run()` implements the swarms Agent interface, accepting dict or JSON
# string inputs and returning structured results compatible with swarms workflows.
