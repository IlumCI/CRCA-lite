

from typing import Dict, Any, List, Tuple, Optional, Union
import asyncio
import logging
import math
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

# Optional heavy dependencies — used when available
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats  # type: ignore
    from scipy import linalg as scipy_linalg  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import cvxpy as cp  # type: ignore
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    import importlib
    lu_spec = importlib.util.find_spec("litellm.litellm_core_utils.logging_utils")
    if lu_spec is not None:
        lu = importlib.import_module("litellm.litellm_core_utils.logging_utils")
        try:
            if hasattr(lu, "asyncio") and hasattr(lu.asyncio, "iscoroutinefunction"):
                lu.asyncio.iscoroutinefunction = inspect.iscoroutinefunction
        except Exception:
            pass
except Exception:
    pass


class CausalRelationType(Enum):
    
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class CausalNode:
    
    name: str
    value: Optional[float] = None
    confidence: float = 1.0
    node_type: str = "variable"


@dataclass
class CausalEdge:
    
    source: str
    target: str
    strength: float = 1.0
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    confidence: float = 1.0


@dataclass
class CounterfactualScenario:
    
    name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float = 1.0
    reasoning: str = ""


class CRCAAgent(Agent):
    

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "cr-ca-lite-agent",
        agent_description: str = "Lightweight Causal Reasoning with Counterfactual Analysis Agent",
        description: Optional[str] = None,
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        enable_batch_predict: bool = False,
        max_batch_size: int = 32,
        bootstrap_workers: int = 0,
        use_async: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        
        cr_ca_schema = CRCAAgent._get_cr_ca_schema()
        
        # Backwards-compatible alias for description
        agent_description = description or agent_description

        agent_kwargs = {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "model_name": model_name,
            "max_loops": 1,  # Individual LLM calls use 1 loop, we handle multi-loop reasoning
            "tools_list_dictionary": [cr_ca_schema],
            "output_type": "final",
            **kwargs,
        }
        
        if system_prompt is not None:
            agent_kwargs["system_prompt"] = system_prompt
        if global_system_prompt is not None:
            agent_kwargs["global_system_prompt"] = global_system_prompt
        if secondary_system_prompt is not None:
            agent_kwargs["secondary_system_prompt"] = secondary_system_prompt
        
        super().__init__(**agent_kwargs)
        
        self.causal_max_loops = max_loops
        self.causal_graph: Dict[str, Dict[str, float]] = {}
        self.causal_graph_reverse: Dict[str, List[str]] = {}  # For fast parent lookup
        self._graph = rx.PyDiGraph()
        self._node_to_index: Dict[str, int] = {}
        self._index_to_node: Dict[int, str] = {}
        
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        self.use_nonlinear_scm: bool = True
        self.nonlinear_activation: str = "tanh"  # options: 'tanh'|'identity'
        self.interaction_terms: Dict[str, List[Tuple[str, str]]] = {}
        self.edge_sign_constraints: Dict[Tuple[str, str], int] = {}
        self.bayesian_priors: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.enable_batch_predict = bool(enable_batch_predict)
        self.max_batch_size = int(max_batch_size)
        self.bootstrap_workers = int(max(0, bootstrap_workers))
        self.use_async = bool(use_async)
        self.seed = seed if seed is not None else 42
        self._rng = np.random.default_rng(self.seed)
        
        if variables:
            for var in variables:
                self._ensure_node_exists(var)

        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

        self.causal_memory: List[Dict[str, Any]] = []
        self._prediction_cache: Dict[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]], Dict[str, float]] = {}
        self._prediction_cache_order: List[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = []
        self._prediction_cache_max: int = 1000
        self._cache_enabled: bool = True
        self._prediction_cache_lock = threading.Lock()

    @staticmethod
    def _get_cr_ca_schema() -> Dict[str, Any]:
        
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
        
        response = super().run(task)
        return response

    def _build_causal_prompt(self, task: str) -> str:
        return (
            "You are a Causal Reasoning with Counterfactual Analysis (CR-CA) agent.\n"
            f"Problem: {task}\n"
            f"Current causal graph has {len(self.causal_graph)} variables and "
            f"{sum(len(children) for children in self.causal_graph.values())} relationships.\n"
        )

    def _build_memory_context(self) -> str:
        
        context_parts = []
        for step in self.causal_memory[-2:]:  # Last 2 steps
            context_parts.append(f"Step {step['step']}: {step['analysis']}")
        return "\n".join(context_parts)

    def _synthesize_causal_analysis(self, task: str) -> str:
        
        synthesis_prompt = f"Based on the causal analysis steps performed, synthesize a concise causal report for: {task}"
        return self.step(synthesis_prompt)
    def _ensure_node_exists(self, node: str) -> None:
        
        if node not in self.causal_graph:
            self.causal_graph[node] = {}
        if node not in self.causal_graph_reverse:
            self.causal_graph_reverse[node] = []
        try:
            self._ensure_node_index(node)
        except Exception:
            pass

    def add_causal_relationship(
        self, 
        source: str, 
        target: str, 
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)

        meta = {
            "strength": float(strength),
            "relation_type": relation_type.value if isinstance(relation_type, Enum) else str(relation_type),
            "confidence": float(confidence),
        }

        self.causal_graph.setdefault(source, {})[target] = meta

        if source not in self.causal_graph_reverse.get(target, []):
            self.causal_graph_reverse.setdefault(target, []).append(source)

        try:
            u_idx = self._ensure_node_index(source)
            v_idx = self._ensure_node_index(target)
            try:
                existing = self._graph.get_edge_data(u_idx, v_idx)
            except Exception:
                existing = None

            if existing is None:
                try:
                    self._graph.add_edge(u_idx, v_idx, meta)
                except Exception:
                    try:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"rustworkx.add_edge failed for {source}->{target}; continuing with dict-only graph."
                        )
                    except Exception:
                        pass
            else:
                try:
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
            try:
                import logging
                logging.getLogger(__name__).warning(
                    "rustworkx operation failed during add_causal_relationship; continuing with dict-only graph."
                )
            except Exception:
                pass
    
    def _get_parents(self, node: str) -> List[str]:
        
        return self.causal_graph_reverse.get(node, [])
    
    def _get_children(self, node: str) -> List[str]:
        
        return list(self.causal_graph.get(node, {}).keys())

    def _ensure_node_index(self, name: str) -> int:
        
        if name in self._node_to_index:
            return self._node_to_index[name]
        idx = self._graph.add_node(name)
        self._node_to_index[name] = idx
        self._index_to_node[idx] = name
        return idx

    def _node_index(self, name: str) -> Optional[int]:
        
        return self._node_to_index.get(name)

    def _node_name(self, idx: int) -> Optional[str]:
        
        return self._index_to_node.get(idx)

    def _edge_strength(self, source: str, target: str) -> float:
        
        edge = self.causal_graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0
    
    def _topological_sort(self) -> List[str]:
        
        try:
            order_idx = rx.topological_sort(self._graph)
            result = [self._node_name(i) for i in order_idx if self._node_name(i) is not None]
            for n in list(self.causal_graph.keys()):
                if n not in result:
                    result.append(n)
            return result
        except Exception:
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
        
        if start not in self.causal_graph or end not in self.causal_graph:
            return []
        
        if start == end:
            return [start]
        
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: set = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            for child in self._get_children(current):
                if child == end:
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []  # No path found
    
    
    def _has_path(self, start: str, end: str) -> bool:
        
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
        
        with self._prediction_cache_lock:
            self._prediction_cache.clear()
            self._prediction_cache_order.clear()

    def enable_cache(self, flag: bool) -> None:
        
        with self._prediction_cache_lock:
            self._cache_enabled = bool(flag)
    
    
    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        
        z: Dict[str, float] = {}
        for k, v in state.items():
            s = self.standardization_stats.get(k)
            if s and s.get("std", 0.0) > 0:
                z[k] = (v - s["mean"]) / s["std"]
            else:
                z[k] = v
        return z
    
    def _destandardize_value(self, var: str, z_value: float) -> float:
        
        s = self.standardization_stats.get(var)
        if s and s.get("std", 0.0) > 0:
            return z_value * s["std"] + s["mean"]
        return z_value
    
    def _predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        
        if self.use_nonlinear_scm:
            z_pred = self._predict_z(factual_state, interventions, use_noise=None)
            return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

        raw = factual_state.copy()
        raw.update(interventions)
        
        z_state = self._standardize_state(raw)
        z_pred = dict(z_state)
        
        for node in self._topological_sort():
            if node in interventions:
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue
            
            parents = self._get_parents(node)
            if not parents:
                continue
            
            s = 0.0
            for p in parents:
                pz = z_pred.get(p, z_state.get(p, 0.0))
                strength = self._edge_strength(p, node)
                s += pz * strength
            
            z_pred[node] = s
        
        return {v: self._destandardize_value(v, z) for v, z in z_pred.items()}

    def _predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self._predict_outcomes(factual_state, interventions)

        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        result = self._predict_outcomes(factual_state, interventions)

        with self._prediction_cache_lock:
            if len(self._prediction_cache_order) >= self._prediction_cache_max:
                remove_count = max(1, self._prediction_cache_max // 10)
                for _ in range(remove_count):
                    old = self._prediction_cache_order.pop(0)
                    if old in self._prediction_cache:
                        del self._prediction_cache[old]

            self._prediction_cache_order.append(cache_key)
            self._prediction_cache[cache_key] = dict(result)
        return result

    def _get_descendants(self, node: str) -> List[str]:
        
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
                z_val = float(use_noise.get(node, 0.0)) if use_noise else z_state.get(node, 0.0)
                z_pred[node] = z_val
                continue

            linear_term = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_state.get(p, 0.0))
                beta = self._edge_strength(p, node)
                linear_term += parent_z * beta

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

            model_z = linear_term + interaction_term

            if use_noise:
                model_z += float(use_noise.get(node, 0.0))

            if self.nonlinear_activation == "tanh":
                model_z_act = float(np.tanh(model_z) * 3.0)  # scale to limit
            else:
                model_z_act = float(model_z)

            observed_z = z_state.get(node, 0.0)

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
        
        z = self._standardize_state(factual_state)

        z_obs = z  # standardized factual
        pred_no_noise = self._predict_z(factual_state, {}, use_noise=None)
        noise: Dict[str, float] = {}
        for node in pred_no_noise.keys():
            noise[node] = float(z_obs.get(node, 0.0) - pred_no_noise.get(node, 0.0))

        z_pred = self._predict_z(factual_state, interventions, use_noise=noise)
        return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def aap(self, factual_state: Dict[str, float], interventions: Dict[str, float]) -> Dict[str, float]:
        
        return self.counterfactual_abduction_action_prediction(factual_state, interventions)

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        
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
        
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self._predict_outcomes(factual_state, interventions)

        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        result = self._predict_outcomes(factual_state, interventions)

        with self._prediction_cache_lock:
            if len(self._prediction_cache_order) >= self._prediction_cache_max:
                remove_count = max(1, self._prediction_cache_max // 10)
                for _ in range(remove_count):
                    old = self._prediction_cache_order.pop(0)
                    if old in self._prediction_cache:
                        del self._prediction_cache[old]

            self._prediction_cache_order.append(cache_key)
            self._prediction_cache[cache_key] = dict(result)
        return result

    def _get_descendants(self, node: str) -> List[str]:
        
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
        
        z = self._standardize_state(factual_state)

        noise: Dict[str, float] = {}
        for node in self._topological_sort():
            parents = self._get_parents(node)
            if not parents:
                noise[node] = float(z.get(node, 0.0))
                continue

            pred_z = 0.0
            for p in parents:
                pz = z.get(p, 0.0)
                strength = self._edge_strength(p, node)
                pred_z += pz * strength

            noise[node] = float(z.get(node, 0.0) - pred_z)

        cf_raw = factual_state.copy()
        cf_raw.update(interventions)
        z_cf = self._standardize_state(cf_raw)

        z_pred: Dict[str, float] = {}
        for node in self._topological_sort():
            if node in interventions:
                z_pred[node] = float(z_cf.get(node, 0.0))
                continue

            parents = self._get_parents(node)
            if not parents:
                z_pred[node] = float(noise.get(node, 0.0))
                continue

            val = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_cf.get(p, 0.0))
                strength = self._edge_strength(p, node)
                val += parent_z * strength

            z_pred[node] = float(val + noise.get(node, 0.0))

        return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        
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
        
        self.ensure_standardization_stats(factual_state)

        scenarios: List[CounterfactualScenario] = []
        z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        for i, tv in enumerate(target_variables[:max_scenarios]):
            stats = self.standardization_stats.get(tv, {"mean": 0.0, "std": 1.0})
            cur = factual_state.get(tv, stats.get("mean", 0.0))

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
        
        self.standardization_stats[variable] = {"mean": mean, "std": std if std > 0 else 1.0}
    
    def ensure_standardization_stats(self, state: Dict[str, float]) -> None:
        
        for var, val in state.items():
            if var not in self.standardization_stats:
                self.standardization_stats[var] = {"mean": float(val), "std": 1.0}
    
    def get_nodes(self) -> List[str]:
        
        return list(self.causal_graph.keys())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        
        edges = []
        for source, targets in self.causal_graph.items():
            for target in targets.keys():
                edges.append((source, target))
        return edges
    
    def is_dag(self) -> bool:
        
        try:
            return rx.is_directed_acyclic_graph(self._graph)
        except Exception:
            def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
                
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
        
        if task is not None and isinstance(task, str) and initial_state is None and not task.strip().startswith('{'):
            return self._run_llm_causal_analysis(task, **kwargs)
        
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
            target_variables = list(self.causal_graph.keys())
        
        def _resolve_max_steps(value: Union[int, str]) -> int:
            if isinstance(value, str) and value == "auto":
                return max(1, len(self.causal_graph))
            try:
                return int(value)
            except Exception:
                return max(1, len(self.causal_graph))

        effective_steps = _resolve_max_steps(max_steps if max_steps != 1 or self.causal_max_loops == 1 else self.causal_max_loops)
        if max_steps == 1 and self.causal_max_loops != 1:
            effective_steps = _resolve_max_steps(self.causal_max_loops)

        current_state = initial_state.copy()
        for step in range(effective_steps):
            current_state = self._predict_outcomes(current_state, {})
        
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
        
        self.causal_memory = []
        
        causal_prompt = self._build_causal_prompt(task)
        
        max_loops = self.causal_max_loops if isinstance(self.causal_max_loops, int) else 3
        for i in range(max_loops):
            step_result = self.step(causal_prompt)
            self.causal_memory.append({
                'step': i + 1,
                'analysis': step_result,
                'timestamp': i
            })
            
            if i < max_loops - 1:
                memory_context = self._build_memory_context()
                causal_prompt = f"{causal_prompt}\n\nPrevious Analysis:\n{memory_context}"
        
        final_analysis = self._synthesize_causal_analysis(task)
        
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

    # =========================
    # Compatibility extensions
    # =========================

    # ---- Helpers ----
    @staticmethod
    def _require_pandas() -> None:
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for this operation. Install pandas to proceed.")

    @staticmethod
    def _require_scipy() -> None:
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for this operation. Install scipy to proceed.")

    @staticmethod
    def _require_cvxpy() -> None:
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for this operation. Install cvxpy to proceed.")

    def _edge_strength(self, source: str, target: str) -> float:
        edge = self.causal_graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0

    class _TimeDebug:
        def __init__(self, name: str) -> None:
            self.name = name
            self.start = 0.0
        def __enter__(self):
            try:
                import time
                self.start = time.perf_counter()
            except Exception:
                self.start = 0.0
            return self
        def __exit__(self, exc_type, exc, tb):
            if logger.isEnabledFor(logging.DEBUG) and self.start:
                try:
                    import time
                    duration = time.perf_counter() - self.start
                    logger.debug(f"{self.name} completed in {duration:.4f}s")
                except Exception:
                    pass

    def _ensure_edge(self, source: str, target: str) -> None:
        """Ensure edge exists in both dict graph and rustworkx graph."""
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)
        if target not in self.causal_graph.get(source, {}):
            self.causal_graph.setdefault(source, {})[target] = {"strength": 0.0, "confidence": 1.0}
            try:
                u_idx = self._ensure_node_index(source)
                v_idx = self._ensure_node_index(target)
                if self._graph.get_edge_data(u_idx, v_idx) is None:
                    self._graph.add_edge(u_idx, v_idx, self.causal_graph[source][target])
            except Exception:
                pass

    # Convenience graph-like methods for compatibility
    def add_nodes_from(self, nodes: List[str]) -> None:
        for n in nodes:
            self._ensure_node_exists(n)

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> None:
        for u, v in edges:
            self.add_causal_relationship(u, v)

    def edges(self) -> List[Tuple[str, str]]:
        return self.get_edges()

    # ---- Batched predictions ----
    def _predict_outcomes_batch(
        self,
        factual_states: List[Dict[str, float]],
        interventions: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
    ) -> List[Dict[str, float]]:
        """
        Batched deterministic SCM forward pass. Uses shared topology and vectorized parent aggregation.
        """
        if not factual_states:
            return []
        if len(factual_states) == 1 or not self.enable_batch_predict:
            return [self._predict_outcomes(factual_states[0], interventions if isinstance(interventions, dict) else (interventions or {}))]

        batch = len(factual_states)
        if interventions is None:
            interventions_list = [{} for _ in range(batch)]
        elif isinstance(interventions, list):
            interventions_list = interventions
        else:
            interventions_list = [interventions for _ in range(batch)]

        topo = self._topological_sort()
        parents_map = {node: self._get_parents(node) for node in topo}
        stats = self.standardization_stats
        z_pred: Dict[str, np.ndarray] = {}

        # Initialize z with raw + interventions standardized
        for node in topo:
            arr = np.empty(batch, dtype=float)
            mean = stats.get(node, {}).get("mean", 0.0)
            std = stats.get(node, {}).get("std", 1.0) or 1.0
            for i in range(batch):
                raw_val = interventions_list[i].get(node, factual_states[i].get(node, 0.0))
                arr[i] = (raw_val - mean) / std
            z_pred[node] = arr

        # Propagate for non-intervened nodes
        for node in topo:
            parents = parents_map.get(node, [])
            if not parents:
                continue
            arr = z_pred[node]
            # Only recompute if node not directly intervened
            intervene_mask = np.array([node in interventions_list[i] for i in range(batch)], dtype=bool)
            if np.all(intervene_mask):
                continue
            if not parents:
                continue
            parent_matrix = np.vstack([z_pred[p] for p in parents])  # shape (k, batch)
            strengths = np.array([self._edge_strength(p, node) for p in parents], dtype=float).reshape(-1, 1)
            combined = (strengths * parent_matrix).sum(axis=0)
            if intervene_mask.any():
                # preserve intervened samples
                arr = np.where(intervene_mask, arr, combined)
            else:
                arr = combined
            z_pred[node] = arr

        # De-standardize
        outputs: List[Dict[str, float]] = []
        for i in range(batch):
            out: Dict[str, float] = {}
            for node, z_arr in z_pred.items():
                s = stats.get(node, {"mean": 0.0, "std": 1.0})
                out[node] = float(z_arr[i] * s.get("std", 1.0) + s.get("mean", 0.0))
            outputs.append(out)
        return outputs

    # Convenience graph-like methods for compatibility
    def add_nodes_from(self, nodes: List[str]) -> None:
        for n in nodes:
            self._ensure_node_exists(n)

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> None:
        for u, v in edges:
            self.add_causal_relationship(u, v)

    def edges(self) -> List[Tuple[str, str]]:
        return self.get_edges()

    # ---- Data-driven fitting ----
    def fit_from_dataframe(
        self,
        df: Any,
        variables: List[str],
        window: int = 30,
        decay_alpha: float = 0.9,
        ridge_lambda: float = 0.0,
        enforce_signs: bool = True
    ) -> None:
        """
        Fit edge strengths and standardization stats from a rolling window with recency weighting.
        """
        with self._TimeDebug("fit_from_dataframe"):
            self._require_pandas()
            if df is None:
                return
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
            if not variables:
                return
            missing = [v for v in variables if v not in df.columns]
            if missing:
                raise ValueError(f"Variables not in DataFrame: {missing}")
            window = max(1, int(window))
            if not (0 < decay_alpha <= 1):
                raise ValueError("decay_alpha must be in (0,1]")

            df_local = df[variables].dropna().copy()
            if df_local.empty:
                return
            window_df = df_local.tail(window)
            n = len(window_df)
            weights = np.array([decay_alpha ** (n - 1 - i) for i in range(n)], dtype=float)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

            # Standardization stats
            self.standardization_stats = {}
            for v in variables:
                m = float(window_df[v].mean())
                s = float(window_df[v].std(ddof=0))
                if s == 0:
                    s = 1.0
                self.standardization_stats[v] = {"mean": m, "std": s}
            for node in self.causal_graph.keys():
                if node not in self.standardization_stats:
                    self.standardization_stats[node] = {"mean": 0.0, "std": 1.0}

            # Estimate edge strengths
            for child in list(self.causal_graph.keys()):
                parents = self._get_parents(child)
                if not parents:
                    continue
                if child not in window_df.columns:
                    continue
                parent_vals = []
                for p in parents:
                    if p in window_df.columns:
                        stats = self.standardization_stats.get(p, {"mean": 0.0, "std": 1.0})
                        parent_vals.append(((window_df[p] - stats["mean"]) / stats["std"]).values)
                if not parent_vals:
                    continue
                X = np.vstack(parent_vals).T
                y_stats = self.standardization_stats.get(child, {"mean": 0.0, "std": 1.0})
                y = ((window_df[child] - y_stats["mean"]) / y_stats["std"]).values
                W = np.diag(weights)
                XtW = X.T @ W
                XtWX = XtW @ X
                if ridge_lambda > 0 and XtWX.size > 0:
                    k = XtWX.shape[0]
                    XtWX = XtWX + ridge_lambda * np.eye(k)
                try:
                    XtWX_inv = np.linalg.pinv(XtWX)
                    beta = XtWX_inv @ (XtW @ y)
                except Exception:
                    beta = np.zeros(X.shape[1])
                beta = np.asarray(beta)
                for idx, p in enumerate(parents):
                    strength = float(beta[idx]) if idx < len(beta) else 0.0
                    if enforce_signs:
                        sign = self.edge_sign_constraints.get((p, child))
                        if sign == 1 and strength < 0:
                            strength = 0.0
                        elif sign == -1 and strength > 0:
                            strength = 0.0
                    self._ensure_edge(p, child)
                    self.causal_graph[p][child]["strength"] = strength
                    self.causal_graph[p][child]["confidence"] = 1.0

    # ---- Uncertainty ----
    def quantify_uncertainty(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        with self._TimeDebug("quantify_uncertainty"):
            self._require_pandas()
            if df is None or not isinstance(df, pd.DataFrame):
                return {"edge_cis": {}, "samples": 0}
            usable = df[variables].dropna()
            if len(usable) < 10:
                return {"edge_cis": {}, "samples": 0}
            windows = max(1, int(windows))
            samples: Dict[Tuple[str, str], List[float]] = {}

        # Snapshot current strengths to restore later
        baseline_strengths: Dict[Tuple[str, str], float] = {}
        for u, targets in self.causal_graph.items():
            for v, meta in targets.items():
                try:
                    baseline_strengths[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                except Exception:
                    baseline_strengths[(u, v)] = 0.0
        baseline_stats = dict(self.standardization_stats)

        def _snapshot_strengths() -> Dict[Tuple[str, str], float]:
            snap: Dict[Tuple[str, str], float] = {}
            for u, targets in self.causal_graph.items():
                for v, meta in targets.items():
                    try:
                        snap[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        snap[(u, v)] = 0.0
            return snap

        def _bootstrap_single(df_sample: "pd.DataFrame") -> Dict[Tuple[str, str], float]:
            # Use a shallow clone to avoid mutating main agent when running in parallel
            clone = CRCAAgent(
                variables=list(self.causal_graph.keys()),
                causal_edges=self.get_edges(),
                model_name=self.model_name,
                max_loops=self.causal_max_loops,
                enable_batch_predict=self.enable_batch_predict,
                max_batch_size=self.max_batch_size,
                bootstrap_workers=0,
                use_async=self.use_async,
                seed=self.seed,
            )
            clone.edge_sign_constraints = dict(self.edge_sign_constraints)
            clone.standardization_stats = dict(baseline_stats)
            try:
                clone.fit_from_dataframe(
                    df=df_sample,
                    variables=variables,
                    window=min(30, len(df_sample)),
                    decay_alpha=0.9,
                    ridge_lambda=0.0,
                    enforce_signs=True,
                )
                return _snapshot_strengths_from_graph(clone.causal_graph)
            except Exception:
                return {}

        def _snapshot_strengths_from_graph(graph: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
            res: Dict[Tuple[str, str], float] = {}
            for u, targets in graph.items():
                for v, meta in targets.items():
                    try:
                        res[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        res[(u, v)] = 0.0
            return res

        use_parallel = self.bootstrap_workers > 0
        if use_parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.bootstrap_workers) as executor:
                futures = []
                for i in range(windows):
                    boot_df = usable.sample(n=len(usable), replace=True, random_state=self.seed + i)
                    futures.append(executor.submit(_bootstrap_single, boot_df))
                for fut in futures:
                    try:
                        res_strengths = fut.result()
                        for (u, v), w in res_strengths.items():
                            samples.setdefault((u, v), []).append(w)
                    except Exception:
                        continue
        else:
            for i in range(windows):
                boot_df = usable.sample(n=len(usable), replace=True, random_state=self.seed + i)
                try:
                    self.fit_from_dataframe(
                        df=boot_df,
                        variables=variables,
                        window=min(30, len(boot_df)),
                        decay_alpha=0.9,
                        ridge_lambda=0.0,
                        enforce_signs=True,
                    )
                    for (u, v), w in _snapshot_strengths().items():
                        samples.setdefault((u, v), []).append(w)
                except Exception:
                    continue

        # Restore baseline strengths and stats
        for (u, v), w in baseline_strengths.items():
            if u in self.causal_graph and v in self.causal_graph[u]:
                self.causal_graph[u][v]["strength"] = w
        self.standardization_stats = baseline_stats

        edge_cis: Dict[str, Tuple[float, float]] = {}
        for (u, v), arr in samples.items():
            arr_np = np.array(arr)
            lo = float(np.quantile(arr_np, (1 - alpha) / 2))
            hi = float(np.quantile(arr_np, 1 - (1 - alpha) / 2))
            edge_cis[f"{u}->{v}"] = (lo, hi)
        return {"edge_cis": edge_cis, "samples": windows}

    # ---- Optimization ----
    def gradient_based_intervention_optimization(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, Any]:
        self._require_scipy()
        from scipy.optimize import minimize  # type: ignore

        if not intervention_vars:
            return {"error": "intervention_vars cannot be empty", "optimal_intervention": {}, "success": False}

        bounds = []
        x0 = []
        for var in intervention_vars:
            cur = float(initial_state.get(var, 0.0))
            x0.append(cur)
            if constraints and var in constraints:
                bounds.append(constraints[var])
            else:
                bounds.append((cur - 3.0, cur + 3.0))

        def objective(x: np.ndarray) -> float:
            intervention = {intervention_vars[i]: float(x[i]) for i in range(len(x))}
            outcome = self._predict_outcomes(initial_state, intervention)
            return -float(outcome.get(target, 0.0))

        try:
            result = minimize(
                objective,
                x0=np.array(x0, dtype=float),
                method=method,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-6},
            )
            optimal_intervention = {intervention_vars[i]: float(result.x[i]) for i in range(len(result.x))}
            optimal_outcome = self._predict_outcomes(initial_state, optimal_intervention)
            return {
                "optimal_intervention": optimal_intervention,
                "optimal_target_value": float(optimal_outcome.get(target, 0.0)),
                "objective_value": float(result.fun),
                "success": bool(result.success),
                "iterations": int(getattr(result, "nit", 0)),
                "convergence_message": str(result.message),
            }
        except Exception as e:
            logger.debug(f"gradient_based_intervention_optimization failed: {e}")
            return {"error": str(e), "optimal_intervention": {}, "success": False}

    def bellman_optimal_intervention(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        horizon: int = 5,
        discount: float = 0.9,
    ) -> Dict[str, Any]:
        if not intervention_vars:
            return {"error": "intervention_vars cannot be empty"}
        horizon = max(1, int(horizon))
        rng = np.random.default_rng(self.seed)
        current_state = dict(initial_state)
        sequence: List[Dict[str, float]] = []

        def reward(state: Dict[str, float]) -> float:
            return float(state.get(target, 0.0))

        for _ in range(horizon):
            best_value = float("-inf")
            best_intervention: Dict[str, float] = {}
            for _ in range(10):
                candidate = {}
                for var in intervention_vars:
                    stats = self.standardization_stats.get(var, {"mean": current_state.get(var, 0.0), "std": 1.0})
                    candidate[var] = float(rng.normal(stats["mean"], stats["std"]))
                next_state = self._predict_outcomes(current_state, candidate)
                val = reward(next_state)
                if val > best_value:
                    best_value = val
                    best_intervention = candidate
            if best_intervention:
                sequence.append(best_intervention)
                current_state = self._predict_outcomes(current_state, best_intervention)

        return {
            "optimal_sequence": sequence,
            "final_state": current_state,
            "total_value": float(current_state.get(target, 0.0)),
            "horizon": horizon,
            "discount_factor": float(discount),
        }

    # ---- Time-series & causality ----
    def granger_causality_test(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data or variables"}
        data = df[[var1, var2]].dropna()
        if len(data) < max_lag * 2 + 5:
            return {"error": "Insufficient data"}
        try:
            from scipy.stats import f as f_dist  # type: ignore
        except Exception:
            return {"error": "scipy f distribution not available"}

        n = len(data)
        y = data[var2].values
        Xr = []
        Xu = []
        for t in range(max_lag, n):
            y_t = y[t]
            lags_var2 = [data[var2].iloc[t - i] for i in range(1, max_lag + 1)]
            lags_var1 = [data[var1].iloc[t - i] for i in range(1, max_lag + 1)]
            Xr.append(lags_var2)
            Xu.append(lags_var2 + lags_var1)
        y_vec = np.array(y[max_lag:], dtype=float)
        Xr = np.array(Xr, dtype=float)
        Xu = np.array(Xu, dtype=float)

        def ols(X: np.ndarray, yv: np.ndarray) -> Tuple[np.ndarray, float]:
            beta = np.linalg.pinv(X) @ yv
            y_pred = X @ beta
            rss = float(np.sum((yv - y_pred) ** 2))
            return beta, rss

        try:
            _, rss_r = ols(Xr, y_vec)
            _, rss_u = ols(Xu, y_vec)
            m = max_lag
            df2 = len(y_vec) - 2 * m - 1
            if df2 <= 0 or rss_u <= 1e-12:
                return {"error": "Degenerate case in F-test"}
            f_stat = ((rss_r - rss_u) / m) / (rss_u / df2)
            p_value = float(1.0 - f_dist.cdf(f_stat, m, df2))
            return {
                "f_statistic": float(f_stat),
                "p_value": p_value,
                "granger_causes": p_value < 0.05,
                "max_lag": max_lag,
                "restricted_rss": rss_r,
                "unrestricted_rss": rss_u,
            }
        except Exception as e:
            return {"error": str(e)}

    def vector_autoregression_estimation(
        self,
        df: Any,
        variables: List[str],
        max_lag: int = 2,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        data = df[variables].dropna()
        if len(data) < max_lag * len(variables) + 5:
            return {"error": "Insufficient data"}
        n_vars = len(variables)
        X_lag = []
        y_mat = []
        for t in range(max_lag, len(data)):
            y_row = [data[var].iloc[t] for var in variables]
            y_mat.append(y_row)
            lag_row = []
            for lag in range(1, max_lag + 1):
                for var in variables:
                    lag_row.append(data[var].iloc[t - lag])
            X_lag.append(lag_row)
        X = np.array(X_lag, dtype=float)
        Y = np.array(y_mat, dtype=float)
        coefficients: Dict[str, Any] = {}
        residuals = []
        for idx, var in enumerate(variables):
            y_vec = Y[:, idx]
            beta = np.linalg.pinv(X) @ y_vec
            y_pred = X @ beta
            res = y_vec - y_pred
            residuals.append(res)
            coefficients[var] = {"coefficients": beta.tolist()}
        residuals = np.array(residuals).T
        return {
            "coefficient_matrices": coefficients,
            "residuals": residuals.tolist(),
            "n_observations": len(Y),
            "n_variables": n_vars,
            "max_lag": max_lag,
            "variables": variables,
        }

    def compute_information_theoretic_measures(
        self,
        df: Any,
        variables: List[str],
    ) -> Dict[str, Any]:
        """
        Compute simple entropy and mutual information estimates using histograms.
        """
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        data = df[variables].dropna()
        if len(data) < 10:
            return {"error": "Insufficient data"}

        results: Dict[str, Any] = {"entropies": {}, "mutual_information": {}}
        for var in variables:
            if var not in data.columns:
                continue
            series = data[var].dropna()
            if len(series) < 5:
                continue
            n_bins = min(20, max(5, int(np.sqrt(len(series)))))
            hist, _ = np.histogram(series, bins=n_bins)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            results["entropies"][var] = float(entropy)

        # Pairwise mutual information
        for i, var1 in enumerate(variables):
            if var1 not in results["entropies"]:
                continue
            for var2 in variables[i + 1:]:
                if var2 not in results["entropies"]:
                    continue
                joint = data[[var1, var2]].dropna()
                if len(joint) < 5:
                    continue
                n_bins = min(10, max(3, int(np.cbrt(len(joint)))))
                hist2d, _, _ = np.histogram2d(joint[var1], joint[var2], bins=n_bins)
                hist2d = hist2d[hist2d > 0]
                probs_joint = hist2d / hist2d.sum()
                h_joint = -np.sum(probs_joint * np.log2(probs_joint))
                mi = results["entropies"][var1] + results["entropies"][var2] - float(h_joint)
                results["mutual_information"][f"{var1};{var2}"] = float(max(0.0, mi))

        return results

    # ---- Bayesian & attribution ----
    def bayesian_edge_inference(
        self,
        df: Any,
        parent: str,
        child: str,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        if parent not in df.columns or child not in df.columns:
            return {"error": "Variables not found"}
        data = df[[parent, child]].dropna()
        if len(data) < 5:
            return {"error": "Insufficient data"}
        X = data[parent].values.reshape(-1, 1)
        y = data[child].values
        X_mean, X_std = X.mean(), X.std() or 1.0
        y_mean, y_std = y.mean(), y.std() or 1.0
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm
        beta_ols = float((np.linalg.pinv(XtX) @ Xty)[0])
        residuals = y_norm - X_norm @ np.array([beta_ols])
        sigma_sq = float(np.var(residuals))
        tau_likelihood = 1.0 / (sigma_sq + 1e-6)
        tau_prior = 1.0 / (prior_sigma ** 2)
        tau_post = tau_prior + tau_likelihood * len(data)
        mu_post = (tau_prior * prior_mu + tau_likelihood * len(data) * beta_ols) / tau_post
        sigma_post = math.sqrt(1.0 / tau_post)
        ci_lower = mu_post - 1.96 * sigma_post
        ci_upper = mu_post + 1.96 * sigma_post
        self.bayesian_priors[(parent, child)] = {"mu": prior_mu, "sigma": prior_sigma}
        return {
            "posterior_mean": float(mu_post),
            "posterior_std": float(sigma_post),
            "posterior_variance": float(sigma_post ** 2),
            "credible_interval_95": (float(ci_lower), float(ci_upper)),
            "ols_estimate": float(beta_ols),
            "prior_mu": float(prior_mu),
            "prior_sigma": float(prior_sigma),
        }

    def sensitivity_analysis(
        self,
        intervention: Dict[str, float],
        target: str,
        perturbation_size: float = 0.01,
    ) -> Dict[str, Any]:
        base_outcome = self._predict_outcomes({}, intervention)
        base_target = base_outcome.get(target, 0.0)
        sensitivities: Dict[str, float] = {}
        elasticities: Dict[str, float] = {}
        for var, val in intervention.items():
            perturbed = dict(intervention)
            perturbed[var] = val + perturbation_size
            perturbed_outcome = self._predict_outcomes({}, perturbed)
            pert_target = perturbed_outcome.get(target, 0.0)
            sensitivity = (pert_target - base_target) / perturbation_size
            sensitivities[var] = float(sensitivity)
            if abs(base_target) > 1e-6 and abs(val) > 1e-6:
                elasticities[var] = float(sensitivity * (val / base_target))
            else:
                elasticities[var] = 0.0
        most_inf = max(sensitivities.items(), key=lambda x: abs(x[1])) if sensitivities else (None, 0.0)
        total_sens = float(np.linalg.norm(list(sensitivities.values()))) if sensitivities else 0.0
        return {
            "sensitivities": sensitivities,
            "elasticities": elasticities,
            "total_sensitivity": total_sens,
            "most_influential_variable": most_inf[0],
            "most_influential_sensitivity": float(most_inf[1]),
        }

    def deep_root_cause_analysis(
        self,
        problem_variable: str,
        max_depth: int = 20,
        min_path_strength: float = 0.01,
    ) -> Dict[str, Any]:
        if problem_variable not in self.causal_graph:
            return {"error": f"Variable {problem_variable} not in causal graph"}
        all_ancestors = list(self.causal_graph_reverse.get(problem_variable, []))
        root_causes: List[Dict[str, Any]] = []
        paths_to_problem: List[Dict[str, Any]] = []

        def path_strength(path: List[str]) -> float:
            prod = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                prod *= self._edge_strength(u, v)
                if abs(prod) < min_path_strength:
                    return 0.0
            return prod

        for anc in all_ancestors:
            try:
                queue = [(anc, [anc])]
                visited = set()
                while queue:
                    node, path = queue.pop(0)
                    if len(path) - 1 > max_depth:
                        continue
                    if node == problem_variable and len(path) > 1:
                        ps = path_strength(path)
                        if abs(ps) > 0:
                            root_causes.append({
                                "root_cause": path[0],
                                "path_to_problem": path,
                                "path_string": " -> ".join(path),
                                "path_strength": float(ps),
                                "depth": len(path) - 1,
                                "is_exogenous": len(self._get_parents(path[0])) == 0,
                            })
                            paths_to_problem.append({
                                "from": path[0],
                                "to": problem_variable,
                                "path": path,
                                "strength": float(ps),
                            })
                        continue
                    for child in self._get_children(node):
                        if child not in visited:
                            visited.add(child)
                            queue.append((child, path + [child]))
            except Exception:
                continue

        root_causes.sort(key=lambda x: (-x["is_exogenous"], -abs(x["path_strength"]), x["depth"]))
        ultimate_roots = [rc for rc in root_causes if rc.get("is_exogenous")]
        return {
            "problem_variable": problem_variable,
            "all_root_causes": root_causes[:20],
            "ultimate_root_causes": ultimate_roots[:10],
            "total_paths_found": len(paths_to_problem),
            "max_depth_reached": max([rc["depth"] for rc in root_causes] + [0]),
        }

    def shapley_value_attribution(
        self,
        baseline_state: Dict[str, float],
        target_state: Dict[str, float],
        target: str,
        samples: int = 100,
    ) -> Dict[str, Any]:
        variables = list(set(list(baseline_state.keys()) + list(target_state.keys())))
        n = len(variables)
        if n == 0:
            return {"shapley_values": {}, "normalized": {}, "total_attribution": 0.0}
        rng = np.random.default_rng(self.seed)
        contributions: Dict[str, float] = {v: 0.0 for v in variables}

        def value(subset: List[str]) -> float:
            state = dict(baseline_state)
            for var in subset:
                if var in target_state:
                    state[var] = target_state[var]
            outcome = self._predict_outcomes({}, state)
            return float(outcome.get(target, 0.0))

        for _ in range(max(1, samples)):
            perm = list(variables)
            rng.shuffle(perm)
            cur_set: List[str] = []
            prev_val = value(cur_set)
            for v in perm:
                cur_set.append(v)
                new_val = value(cur_set)
                contributions[v] += new_val - prev_val
                prev_val = new_val

        shapley_values = {k: v / float(samples) for k, v in contributions.items()}
        total = sum(abs(v) for v in shapley_values.values()) or 1.0
        normalized = {k: v / total for k, v in shapley_values.items()}
        return {
            "shapley_values": shapley_values,
            "normalized": normalized,
            "total_attribution": float(sum(abs(v) for v in shapley_values.values())),
        }

    # ---- Multi-layer scenarios ----
    def multi_layer_whatif_analysis(
        self,
        scenarios: List[Dict[str, float]],
        depth: int = 3,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for scen in scenarios:
            layer1 = self._predict_outcomes({}, scen)
            affected = [k for k, v in layer1.items() if abs(v) > 0.01]
            layer2_scenarios = [{a: layer1.get(a, 0.0) * 1.2} for a in affected[:5]]
            layer2_results: List[Dict[str, Any]] = []
            for l2 in layer2_scenarios:
                l2_outcome = self._predict_outcomes(layer1, l2)
                layer2_results.append({"layer2_scenario": l2, "layer2_outcomes": l2_outcome})
            results.append({
                "scenario": scen,
                "layer1_direct_effects": layer1,
                "affected_variables": affected,
                "layer2_cascades": layer2_results,
            })
        return {"multi_layer_analysis": results, "summary": {"total_scenarios": len(results)}}

    def explore_alternate_realities(
        self,
        factual_state: Dict[str, float],
        target_outcome: str,
        target_value: Optional[float] = None,
        max_realities: int = 50,
        max_interventions: int = 3,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        variables = list(factual_state.keys())
        realities: List[Dict[str, Any]] = []
        for _ in range(max_realities):
            num_int = rng.integers(1, max(2, max_interventions + 1))
            selected = rng.choice(variables, size=min(num_int, len(variables)), replace=False)
            intervention = {}
            for var in selected:
                stats = self.standardization_stats.get(var, {"mean": factual_state.get(var, 0.0), "std": 1.0})
                intervention[var] = float(rng.normal(stats["mean"], stats["std"] * 1.5))
            outcome = self._predict_outcomes(factual_state, intervention)
            target_val = outcome.get(target_outcome, 0.0)
            if target_value is not None:
                objective = -abs(target_val - target_value)
            else:
                objective = target_val
            realities.append({
                "interventions": intervention,
                "outcome": outcome,
                "target_value": float(target_val),
                "objective": float(objective),
                "delta_from_factual": float(target_val - factual_state.get(target_outcome, 0.0)),
            })
        realities.sort(key=lambda x: x["objective"], reverse=True)
        best = realities[0] if realities else None
        return {
            "factual_state": factual_state,
            "target_outcome": target_outcome,
            "target_value": target_value,
            "best_reality": best,
            "top_10_realities": realities[:10],
            "all_realities_explored": len(realities),
            "improvement_potential": (best["target_value"] - factual_state.get(target_outcome, 0.0)) if best else 0.0,
        }

    # ---- Async wrappers ----
    async def run_async(
        self,
        task: Optional[Union[str, Any]] = None,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(task=task, initial_state=initial_state, target_variables=target_variables, max_steps=max_steps, **kwargs))

    async def quantify_uncertainty_async(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.quantify_uncertainty(df=df, variables=variables, windows=windows, alpha=alpha))

    async def granger_causality_test_async(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.granger_causality_test(df=df, var1=var1, var2=var2, max_lag=max_lag))

    async def vector_autoregression_estimation_async(
        self,
        df: Any,
        variables: List[str],
        max_lag: int = 2,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.vector_autoregression_estimation(df=df, variables=variables, max_lag=max_lag))



