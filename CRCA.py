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
import threading
import inspect
try:
    import rustworkx as rx
except Exception as e:
    raise ImportError("rustworkx is required for the CRCAAgent rustworkx upgrade: pip install rustworkx") from e

# If litellm is installed, patch its logging utils to avoid deprecated asyncio usage
try:
    # litellm's logging utils sometimes call asyncio.iscoroutinefunction which is
    # deprecated on newer Python versions; prefer inspect.iscoroutinefunction.
    import importlib
    lu_spec = importlib.util.find_spec("litellm.litellm_core_utils.logging_utils")
    if lu_spec is not None:
        lu = importlib.import_module("litellm.litellm_core_utils.logging_utils")
        try:
            # Replace the asyncio.iscoroutinefunction reference if present
            if hasattr(lu, "asyncio") and hasattr(lu.asyncio, "iscoroutinefunction"):
                lu.asyncio.iscoroutinefunction = inspect.iscoroutinefunction
        except Exception:
            # If any monkeypatching fails, continue without error.
            pass
except Exception:
    # litellm not present or couldn't be patched — that's fine for deterministic core.
    pass


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
        # rustworkx graph backing (required)
        self._graph = rx.PyDiGraph()
        # node name <-> index maps for rustworkx
        self._node_to_index: Dict[str, int] = {}
        self._index_to_node: Dict[int, str] = {}
        
        # Standardization statistics: {'var': {'mean': m, 'std': s}}
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        # Nonlinear SCM options
        self.use_nonlinear_scm: bool = True
        self.nonlinear_activation: str = "tanh"  # options: 'tanh'|'identity'
        # Interaction terms registry: {child: [(p1,p2), ...]}
        self.interaction_terms: Dict[str, List[Tuple[str, str]]] = {}
        
        # Initialize graph
        if variables:
            for var in variables:
                self._ensure_node_exists(var)

        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

        # Memory for storing causal analysis history (LLM-based)
        self.causal_memory: List[Dict[str, Any]] = []
        # Prediction cache for repeated scenario evaluation (simple FIFO eviction)
        self._prediction_cache: Dict[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]], Dict[str, float]] = {}
        self._prediction_cache_order: List[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = []
        self._prediction_cache_max: int = 1000
        self._cache_enabled: bool = True
        # Lock to protect prediction cache in multithreaded runs
        # Simple coarse-grained lock: protects reads/writes to `_prediction_cache` and `_prediction_cache_order`.
        self._prediction_cache_lock = threading.Lock()
        # (duplicate cache attrs removed)

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
        # Ensure rustworkx node exists and mapping updated
        try:
            self._ensure_node_index(node)
        except Exception:
            # In case rustworkx operations fail unexpectedly, continue without crashing
            pass

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
        # Ensure nodes exist in dict view and rustworkx backing
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)

        # Prepare moderate metadata
        meta = {
            "strength": float(strength),
            "relation_type": relation_type.value if isinstance(relation_type, Enum) else str(relation_type),
            "confidence": float(confidence),
        }

        # Add or update dict view
        self.causal_graph.setdefault(source, {})[target] = meta

        # Update reverse mapping for parent lookup (avoid duplicates)
        if source not in self.causal_graph_reverse.get(target, []):
            self.causal_graph_reverse.setdefault(target, []).append(source)

        # Update rustworkx graph: ensure nodes present and add/update edge data
        try:
            u_idx = self._ensure_node_index(source)
            v_idx = self._ensure_node_index(target)
            # Avoid adding duplicate edges to rustworkx: check existing edge data first.
            try:
                existing = self._graph.get_edge_data(u_idx, v_idx)
            except Exception:
                existing = None

            if existing is None:
                # No existing edge between these nodes — safe to add.
                try:
                    self._graph.add_edge(u_idx, v_idx, meta)
                except Exception:
                    # If add_edge fails, log/debug and continue with dict view
                    try:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"rustworkx.add_edge failed for {source}->{target}; continuing with dict-only graph."
                        )
                    except Exception:
                        pass
            else:
                # Edge already present in rustworkx; attempt to update its metadata in-place.
                try:
                    # If rustworkx returns a dict-like edge data object, update it in-place.
                    if isinstance(existing, dict):
                        existing.update(meta)
                        try:
                            import logging
                            logging.getLogger(__name__).debug(
                                f"Updated rustworkx edge data for {source}->{target} in-place."
                            )
                        except Exception:
                            pass
                    else:
                        # Fallback: try to remove existing edge (by index) and add a new one with updated meta.
                        try:
                            edge_idx = self._graph.get_edge_index(u_idx, v_idx)
                        except Exception:
                            edge_idx = None
                        if edge_idx is not None and edge_idx >= 0:
                            try:
                                self._graph.remove_edge(edge_idx)
                                self._graph.add_edge(u_idx, v_idx, meta)
                                try:
                                    import logging
                                    logging.getLogger(__name__).debug(
                                        f"Replaced rustworkx edge for {source}->{target} with updated metadata."
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    import logging
                                    logging.getLogger(__name__).warning(
                                        f"Could not replace rustworkx edge for {source}->{target}; keeping dict-only metadata."
                                    )
                                except Exception:
                                    pass
                        else:
                            try:
                                import logging
                                logging.getLogger(__name__).debug(
                                    f"rustworkx edge exists but index lookup failed for {source}->{target}; dict metadata used."
                                )
                            except Exception:
                                pass
                except Exception:
                    try:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Failed updating rustworkx edge for {source}->{target}; continuing with dict-only graph."
                        )
                    except Exception:
                        pass
        except Exception:
            # If rustworkx fails unexpectedly, keep dict-only representation
            try:
                import logging
                logging.getLogger(__name__).warning(
                    "rustworkx operation failed during add_causal_relationship; continuing with dict-only graph."
                )
            except Exception:
                pass
    
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

    # --- rustworkx helpers (node index mapping) ---
    def _ensure_node_index(self, name: str) -> int:
        """Ensure a node exists in the rustworkx graph and return its index."""
        if name in self._node_to_index:
            return self._node_to_index[name]
        # Add node with its name as data
        idx = self._graph.add_node(name)
        self._node_to_index[name] = idx
        self._index_to_node[idx] = name
        return idx

    def _node_index(self, name: str) -> Optional[int]:
        """Return rustworkx node index or None."""
        return self._node_to_index.get(name)

    def _node_name(self, idx: int) -> Optional[str]:
        """Return node name for rustworkx index or None."""
        return self._index_to_node.get(idx)

    def _edge_strength(self, source: str, target: str) -> float:
        """Return the numeric strength for edge (source->target) handling metadata or legacy float."""
        edge = self.causal_graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm (pure Python).
        
        Returns:
            List of nodes in topological order
        """
        # Prefer rustworkx topological sort for correctness/performance.
        try:
            order_idx = rx.topological_sort(self._graph)
            # rx.topological_sort may return indices; map back to names
            result = [self._node_name(i) for i in order_idx if self._node_name(i) is not None]
            # Ensure we include any isolated nodes missing from rustworkx order
            for n in list(self.causal_graph.keys()):
                if n not in result:
                    result.append(n)
            return result
        except Exception:
            # Fallback to Kahn's algorithm using dict view
            in_degree: Dict[str, int] = {node: 0 for node in self.causal_graph.keys()}
            for node in self.causal_graph:
                for child in self._get_children(node):
                    in_degree[child] = in_degree.get(child, 0) + 1
            
            queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
            result: List[str] = []
            while queue:
                node = queue.pop(0)
                result.append(node)
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

    def clear_cache(self) -> None:
        """Clear prediction cache and related caches."""
        with self._prediction_cache_lock:
            self._prediction_cache.clear()
            self._prediction_cache_order.clear()

    def enable_cache(self, flag: bool) -> None:
        """Enable or disable prediction caching."""
        with self._prediction_cache_lock:
            self._cache_enabled = bool(flag)
    
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
        # Use nonlinear SCM if enabled, otherwise fall back to linear computation.
        if self.use_nonlinear_scm:
            z_pred = self._predict_z(factual_state, interventions, use_noise=None)
            return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

        # Merge factual state with interventions (linear fallback)
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
                strength = self._edge_strength(p, node)
                s += pz * strength
            
            z_pred[node] = s
        
        # De-standardize results
        return {v: self._destandardize_value(v, z) for v, z in z_pred.items()}

    def _predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Cached wrapper for `_predict_outcomes`. Uses a simple FIFO eviction policy
        when the cache exceeds `self._prediction_cache_max`. Cache keys are based
        on sorted tuples of (var, value) for both factual_state and interventions.
        """
        # Quick check: if caching globally disabled, bypass
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self._predict_outcomes(factual_state, interventions)

        # Build cache keys (sorted for determinism)
        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        # Double-checked locking: check under lock, compute outside, insert under lock
        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        # Compute without holding the cache lock (expensive)
        result = self._predict_outcomes(factual_state, interventions)

        with self._prediction_cache_lock:
            # Evict if necessary (simple FIFO policy)
            if len(self._prediction_cache_order) >= self._prediction_cache_max:
                # Remove oldest 10% to avoid thrashing
                remove_count = max(1, self._prediction_cache_max // 10)
                for _ in range(remove_count):
                    old = self._prediction_cache_order.pop(0)
                    if old in self._prediction_cache:
                        del self._prediction_cache[old]

            self._prediction_cache_order.append(cache_key)
            self._prediction_cache[cache_key] = dict(result)
        return result

    def _get_descendants(self, node: str) -> List[str]:
        """
        Return all descendants of `node` (nodes reachable by following children).
        Uses iterative DFS to avoid recursion limits.
        """
        if node not in self.causal_graph:
            return []
        stack = [node]
        visited = set()
        descendants: List[str] = []
        while stack:
            cur = stack.pop()
            for child in self._get_children(cur):
                if child in visited:
                    continue
                visited.add(child)
                descendants.append(child)
                stack.append(child)
        return descendants

    def _predict_z(self, factual_state: Dict[str, float], interventions: Dict[str, float], use_noise: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Predict in z-space using a nonlinear SCM with optional interaction terms.
        If use_noise is provided, it's a dict of z-noise to add; otherwise noise=0.
        """
        raw = factual_state.copy()
        raw.update(interventions)
        z_state = self._standardize_state(raw)
        z_pred: Dict[str, float] = dict(z_state)

        for node in self._topological_sort():
            if node in interventions:
                z_pred[node] = z_state.get(node, 0.0)
                continue

            parents = self._get_parents(node)
            if not parents:
                # Preserve observed exogenous z-value when no explicit noise is provided.
                # Previously this defaulted to 0.0 which wiped observed variation.
                z_val = float(use_noise.get(node, 0.0)) if use_noise else z_state.get(node, 0.0)
                z_pred[node] = z_val
                continue

            # Linear term
            linear_term = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_state.get(p, 0.0))
                beta = self._edge_strength(p, node)
                linear_term += parent_z * beta

            # Interaction terms (if any)
            interaction_term = 0.0
            for (p1, p2) in self.interaction_terms.get(node, []):
                if p1 in parents and p2 in parents:
                    z1 = z_pred.get(p1, z_state.get(p1, 0.0))
                    z2 = z_pred.get(p2, z_state.get(p2, 0.0))
                    gamma = 0.0
                    edge_data = self.causal_graph.get(p1, {}).get(node, {})
                    if isinstance(edge_data, dict):
                        gamma = float(edge_data.get("interaction_strength", {}).get(p2, 0.0))
                    interaction_term += gamma * z1 * z2

            # Model-implied value (before noise/activation)
            model_z = linear_term + interaction_term

            # Add noise if provided (preserve semantics: noise is additive in z-space)
            if use_noise:
                model_z += float(use_noise.get(node, 0.0))

            # Activation / saturation to prevent explosive linearity
            if self.nonlinear_activation == "tanh":
                model_z_act = float(np.tanh(model_z) * 3.0)  # scale to limit
            else:
                model_z_act = float(model_z)

            # Observed (standardized) value from factual_state (could be a per-step shock)
            observed_z = z_state.get(node, 0.0)

            # Preserve observed shocks only when the observed standardized value itself
            # is materially different from the variable's baseline (mean). This avoids
            # incorrectly preserving children values (like `price`) that were not
            # directly shocked but differ from model due to parent shocks.
            threshold = float(getattr(self, "shock_preserve_threshold", 1e-3))
            if abs(observed_z) > threshold:
                z_pred[node] = float(observed_z)
            else:
                z_pred[node] = float(model_z_act)

        return z_pred

    def counterfactual_abduction_action_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Abduction–Action–Prediction (AAP) for linear SCMs in z-space.
        1) Abduction: infer noise ε from factual_state
        2) Action: apply do(interventions)
        3) Prediction: propagate with preserved noise to get counterfactuals
        """
        # Standardize factual state
        z = self._standardize_state(factual_state)

        # ABDUCTION: infer latent noise using the SCM (predict z without noise then residual)
        z_obs = z  # standardized factual
        pred_no_noise = self._predict_z(factual_state, {}, use_noise=None)
        noise: Dict[str, float] = {}
        for node in pred_no_noise.keys():
            noise[node] = float(z_obs.get(node, 0.0) - pred_no_noise.get(node, 0.0))

        # ACTION + PREDICTION: predict with interventions while preserving abduced noise
        z_pred = self._predict_z(factual_state, interventions, use_noise=noise)
        return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def aap(self, factual_state: Dict[str, float], interventions: Dict[str, float]) -> Dict[str, float]:
        """
        Short alias for `counterfactual_abduction_action_prediction`.
        Use this for concise calls in examples and interactive use.
        """
        return self.counterfactual_abduction_action_prediction(factual_state, interventions)

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        """
        Detect potential confounders: common ancestors of `treatment` and `outcome`.
        Returns a list of variable names that are ancestors of both nodes.
        """
        def _ancestors(node: str) -> set:
            stack = [node]
            visited = set()
            while stack:
                cur = stack.pop()
                for p in self._get_parents(cur):
                    if p in visited:
                        continue
                    visited.add(p)
                    stack.append(p)
            return visited

        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        treat_anc = _ancestors(treatment)
        out_anc = _ancestors(outcome)
        common = treat_anc.intersection(out_anc)
        return list(common)

    def identify_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Heuristic back-door adjustment set: parents of treatment excluding descendants
        of treatment and the outcome itself.
        """
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        parents_t = set(self._get_parents(treatment))
        descendants_t = set(self._get_descendants(treatment))
        adjustment = [z for z in parents_t if z not in descendants_t and z != outcome]
        return adjustment

    def _predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Cached wrapper for `_predict_outcomes`. Uses a simple FIFO eviction policy
        when the cache exceeds `self._prediction_cache_max`. Cache keys are based
        on sorted tuples of (var, value) for both factual_state and interventions.
        """
        # Quick check: if caching globally disabled, bypass
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self._predict_outcomes(factual_state, interventions)

        # Build cache keys (sorted for determinism)
        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        # Double-checked locking: check under lock, compute outside, insert under lock
        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        # Compute without holding the cache lock (expensive)
        result = self._predict_outcomes(factual_state, interventions)

        with self._prediction_cache_lock:
            # Evict if necessary (simple FIFO policy)
            if len(self._prediction_cache_order) >= self._prediction_cache_max:
                # Remove oldest 10% to avoid thrashing
                remove_count = max(1, self._prediction_cache_max // 10)
                for _ in range(remove_count):
                    old = self._prediction_cache_order.pop(0)
                    if old in self._prediction_cache:
                        del self._prediction_cache[old]

            self._prediction_cache_order.append(cache_key)
            self._prediction_cache[cache_key] = dict(result)
        return result

    def _get_descendants(self, node: str) -> List[str]:
        """
        Return all descendants of `node` (nodes reachable by following children).
        Uses iterative DFS to avoid recursion limits.
        """
        if node not in self.causal_graph:
            return []
        stack = [node]
        visited = set()
        descendants: List[str] = []
        while stack:
            cur = stack.pop()
            for child in self._get_children(cur):
                if child in visited:
                    continue
                visited.add(child)
                descendants.append(child)
                stack.append(child)
        return descendants

    def counterfactual_abduction_action_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Abduction–Action–Prediction (AAP) for linear SCMs in z-space.
        1) Abduction: infer noise ε from factual_state
        2) Action: apply do(interventions)
        3) Prediction: propagate with preserved noise to get counterfactuals
        """
        # Standardize factual state
        z = self._standardize_state(factual_state)

        # ABDUCTION: infer latent noise per node (ε = z_observed - Σ β·z_parents)
        noise: Dict[str, float] = {}
        for node in self._topological_sort():
            parents = self._get_parents(node)
            if not parents:
                # Exogenous node: treat observed z as noise (no parents)
                noise[node] = float(z.get(node, 0.0))
                continue

            pred_z = 0.0
            for p in parents:
                pz = z.get(p, 0.0)
                strength = self._edge_strength(p, node)
                pred_z += pz * strength

            noise[node] = float(z.get(node, 0.0) - pred_z)

        # ACTION: build counterfactual standardized inputs (do-operator)
        cf_raw = factual_state.copy()
        cf_raw.update(interventions)
        z_cf = self._standardize_state(cf_raw)

        # PREDICTION: propagate in topological order using abduced noise
        z_pred: Dict[str, float] = {}
        for node in self._topological_sort():
            if node in interventions:
                # Do-operator: force intervened z value
                z_pred[node] = float(z_cf.get(node, 0.0))
                continue

            parents = self._get_parents(node)
            if not parents:
                # Exogenous: use abduced noise
                z_pred[node] = float(noise.get(node, 0.0))
                continue

            val = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_cf.get(p, 0.0))
                strength = self._edge_strength(p, node)
                val += parent_z * strength

            # Add preserved noise
            z_pred[node] = float(val + noise.get(node, 0.0))

        # De-standardize and return
        return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        """
        Detect potential confounders: common ancestors of `treatment` and `outcome`.
        Returns a list of variable names that are ancestors of both nodes.
        """
        def _ancestors(node: str) -> set:
            stack = [node]
            visited = set()
            while stack:
                cur = stack.pop()
                for p in self._get_parents(cur):
                    if p in visited:
                        continue
                    visited.add(p)
                    stack.append(p)
            return visited

        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        treat_anc = _ancestors(treatment)
        out_anc = _ancestors(outcome)
        common = treat_anc.intersection(out_anc)
        return list(common)

    def identify_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Heuristic back-door adjustment set: parents of treatment excluding descendants
        of treatment and the outcome itself.
        """
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        parents_t = set(self._get_parents(treatment))
        descendants_t = set(self._get_descendants(treatment))
        adjustment = [z for z in parents_t if z not in descendants_t and z != outcome]
        return adjustment
    
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
        if source not in self.causal_graph or target not in self.causal_graph.get(source, {}):
            return {"strength": 0.0, "confidence": 0.0, "path_length": float('inf')}
        
        edge = self.causal_graph[source].get(target, {})
        strength = float(edge.get("strength", 0.0)) if isinstance(edge, dict) else float(edge)
        path = self.identify_causal_chain(source, target)
        path_length = len(path) - 1 if path else float('inf')
        
        return {
            "strength": float(strength),
            "confidence": float(edge.get("confidence", 1.0) if isinstance(edge, dict) else 1.0),
            "path_length": path_length,
            "relation_type": edge.get("relation_type", CausalRelationType.DIRECT.value) if isinstance(edge, dict) else CausalRelationType.DIRECT.value
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
        # Prefer rustworkx DAG detection when available
        try:
            return rx.is_directed_acyclic_graph(self._graph)
        except Exception:
            # Fallback to DFS cycle detection on dict view
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
