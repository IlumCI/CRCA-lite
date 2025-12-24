"""
Prediction framework module for specialized agents.

Provides prediction methods, standardization, caching, and counterfactual
scenario generation. Supports both linear and nonlinear prediction models.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import threading
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualScenario:
    """
    Represents a counterfactual scenario with interventions and expected outcomes.
    """
    name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float = 1.0
    reasoning: str = ""


class PredictionFramework:
    """
    Framework for making predictions and generating counterfactual scenarios.
    
    Supports:
    - Standardization/destandardization of variables
    - Linear and nonlinear prediction models
    - Caching of predictions
    - Counterfactual scenario generation
    - Abduction-action prediction (counterfactual reasoning)
    """

    def __init__(
        self,
        graph_manager: Any,  # GraphManager instance
        standardization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        use_nonlinear: bool = True,
        nonlinear_activation: str = "tanh",
        interaction_terms: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
    ):
        """
        Initialize the prediction framework.
        
        Args:
            graph_manager: GraphManager instance for graph operations
            standardization_stats: Dictionary mapping variables to {mean, std}
            use_nonlinear: Whether to use nonlinear prediction model
            nonlinear_activation: Activation function ('tanh' or 'identity')
            interaction_terms: Dictionary mapping nodes to interaction term pairs
            cache_enabled: Whether to enable prediction caching
            cache_max_size: Maximum cache size
        """
        self.graph_manager = graph_manager
        self.standardization_stats = standardization_stats or {}
        self.use_nonlinear_scm = use_nonlinear
        self.nonlinear_activation = nonlinear_activation
        self.interaction_terms = interaction_terms or {}
        self._cache_enabled = cache_enabled
        self._prediction_cache: Dict[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[Tuple[str, float], ...]]], Dict[str, float]] = {}
        self._prediction_cache_order: List[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[Tuple[str, float], ...]]]] = []
        self._prediction_cache_max = cache_max_size
        self._prediction_cache_lock = threading.Lock()

    def standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Standardize a state dictionary using stored statistics.
        
        Args:
            state: Dictionary mapping variables to values
            
        Returns:
            Dictionary mapping variables to standardized values
        """
        z: Dict[str, float] = {}
        for k, v in state.items():
            s = self.standardization_stats.get(k)
            if s and s.get("std", 0.0) > 0:
                z[k] = (v - s["mean"]) / s["std"]
            else:
                z[k] = v
        return z

    def destandardize_value(self, var: str, z_value: float) -> float:
        """
        Destandardize a single value.
        
        Args:
            var: Variable name
            z_value: Standardized value
            
        Returns:
            Destandardized value
        """
        s = self.standardization_stats.get(var)
        if s and s.get("std", 0.0) > 0:
            return z_value * s["std"] + s["mean"]
        return z_value

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
        Ensure standardization stats exist for all variables in state.
        
        Args:
            state: State dictionary
        """
        for var, val in state.items():
            if var not in self.standardization_stats:
                self.standardization_stats[var] = {"mean": float(val), "std": 1.0}

    def predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict outcomes given a factual state and interventions.
        
        Args:
            factual_state: Current state
            interventions: Interventions to apply
            
        Returns:
            Dictionary of predicted outcomes
        """
        if self.use_nonlinear_scm:
            z_pred = self._predict_z(factual_state, interventions, use_noise=None)
            return {v: self.destandardize_value(v, z_val) for v, z_val in z_pred.items()}

        raw = factual_state.copy()
        raw.update(interventions)

        z_state = self.standardize_state(raw)
        z_pred = dict(z_state)

        for node in self.graph_manager.topological_sort():
            if node in interventions:
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue

            parents = self.graph_manager.get_parents(node)
            if not parents:
                continue

            s = 0.0
            for p in parents:
                pz = z_pred.get(p, z_state.get(p, 0.0))
                strength = self.graph_manager.edge_strength(p, node)
                s += pz * strength

            z_pred[node] = s

        return {v: self.destandardize_value(v, z) for v, z in z_pred.items()}

    def _predict_z(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
        use_noise: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Predict outcomes in standardized space.
        
        Args:
            factual_state: Current state
            interventions: Interventions to apply
            use_noise: Optional noise values to add
            
        Returns:
            Dictionary of predicted standardized values
        """
        raw = factual_state.copy()
        raw.update(interventions)
        z_state = self.standardize_state(raw)
        z_pred: Dict[str, float] = dict(z_state)

        for node in self.graph_manager.topological_sort():
            if node in interventions:
                z_pred[node] = z_state.get(node, 0.0)
                continue

            parents = self.graph_manager.get_parents(node)
            if not parents:
                z_val = float(use_noise.get(node, 0.0)) if use_noise else z_state.get(node, 0.0)
                z_pred[node] = z_val
                continue

            linear_term = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_state.get(p, 0.0))
                beta = self.graph_manager.edge_strength(p, node)
                linear_term += parent_z * beta

            interaction_term = 0.0
            for (p1, p2) in self.interaction_terms.get(node, []):
                if p1 in parents and p2 in parents:
                    z1 = z_pred.get(p1, z_state.get(p1, 0.0))
                    z2 = z_pred.get(p2, z_state.get(p2, 0.0))
                    gamma = 0.0
                    edge_data = self.graph_manager.graph.get(p1, {}).get(node, {})
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

    def predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Predict outcomes with caching.
        
        Args:
            factual_state: Current state
            interventions: Interventions to apply
            
        Returns:
            Dictionary of predicted outcomes
        """
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self.predict_outcomes(factual_state, interventions)

        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        result = self.predict_outcomes(factual_state, interventions)

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

    def counterfactual_abduction_action_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Perform counterfactual abduction-action prediction.
        
        This method preserves the noise from the factual state when making
        counterfactual predictions, enabling proper counterfactual reasoning.
        
        Args:
            factual_state: The factual (observed) state
            interventions: Interventions to apply in the counterfactual
            
        Returns:
            Dictionary of counterfactual predictions
        """
        z = self.standardize_state(factual_state)

        noise: Dict[str, float] = {}
        for node in self.graph_manager.topological_sort():
            parents = self.graph_manager.get_parents(node)
            if not parents:
                noise[node] = float(z.get(node, 0.0))
                continue

            pred_z = 0.0
            for p in parents:
                pz = z.get(p, 0.0)
                strength = self.graph_manager.edge_strength(p, node)
                pred_z += pz * strength

            noise[node] = float(z.get(node, 0.0) - pred_z)

        cf_raw = factual_state.copy()
        cf_raw.update(interventions)
        z_cf = self.standardize_state(cf_raw)

        z_pred: Dict[str, float] = {}
        for node in self.graph_manager.topological_sort():
            if node in interventions:
                z_pred[node] = float(z_cf.get(node, 0.0))
                continue

            parents = self.graph_manager.get_parents(node)
            if not parents:
                z_pred[node] = float(noise.get(node, 0.0))
                continue

            val = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_cf.get(p, 0.0))
                strength = self.graph_manager.edge_strength(p, node)
                val += parent_z * strength

            z_pred[node] = float(val + noise.get(node, 0.0))

        return {v: self.destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def calculate_scenario_probability(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> float:
        """
        Calculate the probability of a counterfactual scenario.
        
        Args:
            factual_state: Current state
            interventions: Proposed interventions
            
        Returns:
            Probability value between 0.05 and 0.98
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
        
        Args:
            factual_state: Current state
            target_variables: Variables to generate scenarios for
            max_scenarios: Maximum number of scenarios per variable
            
        Returns:
            List of CounterfactualScenario objects
        """
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
                        expected_outcomes=self.predict_outcomes(
                            factual_state, interventions
                        ),
                        probability=self.calculate_scenario_probability(
                            factual_state, interventions
                        ),
                        reasoning=f"Intervention on {tv} with value {v}",
                    )
                )

        return scenarios

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

