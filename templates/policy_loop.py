"""Policy loop mixin for temporal policy engine.

This module provides PolicyLoopMixin for executing temporal policy loops with:
- Online learning (RLS + Bayesian regression)
- Multi-objective constrained optimization
- Drift detection (CUSUM)
- Deterministic decision-making

This is a feature mixin that can be composed into agent classes (R1 requirement).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from loguru import logger

from schemas.policy import (
    DoctrineV1,
    CompiledPolicy,
    ObservationEvent,
    DecisionEvent,
    OutcomeEvent,
    InterventionSpec,
    Intervention,
    ModelState,
    LedgerEvent
)
from utils.canonical import stable_hash
from utils.ledger import Ledger

# Try to import scipy for numerical stability
try:
    from scipy import linalg as scipy_linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - using numpy for matrix operations")

# Try to import sensors/actuators
try:
    from tools.sensors import SensorRegistry
    from tools.actuators import ActuatorRegistry
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    SensorRegistry = None
    ActuatorRegistry = None
    logger.warning("tools.sensors/actuators not available - using callable functions only")

# Try to import MPC planner
try:
    from templates.mpc_planner import MPCPlanner
    MPC_AVAILABLE = True
except ImportError:
    MPC_AVAILABLE = False
    MPCPlanner = None
    logger.warning("templates.mpc_planner not available - MPC planning disabled")

# Try to import drift detection
try:
    from templates.drift_detection import DriftDetector, HybridDriftDetector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False
    DriftDetector = None
    HybridDriftDetector = None
    logger.warning("templates.drift_detection not available - using CUSUM only")

# Try to import rollback manager
try:
    from utils.rollback import RollbackManager
    ROLLBACK_AVAILABLE = True
except ImportError:
    ROLLBACK_AVAILABLE = False
    RollbackManager = None
    logger.warning("utils.rollback not available - rollback functionality disabled")


class InterventionCatalog:
    """Registry mapping lever IDs to intervention constructors.
    
    Provides deterministic intervention creation with bounds enforcement.
    """
    
    def __init__(self, doctrine: DoctrineV1):
        """
        Initialize catalog from doctrine.
        
        Args:
            doctrine: Doctrine containing lever specifications
        """
        self.doctrine = doctrine
        self.constructors: Dict[str, Callable] = {}
        self._register_default_constructors()
    
    def _register_default_constructors(self) -> None:
        """Register default intervention constructors."""
        for lever_id, lever_spec in self.doctrine.levers.items():
            self.constructors[lever_id] = self._create_constructor(lever_id, lever_spec)
    
    def _create_constructor(self, lever_id: str, lever_spec) -> Callable:
        """Create a constructor function for a lever."""
        def constructor(**params) -> InterventionSpec:
            # Validate bounds
            bounds = lever_spec.bounds
            for param_name, param_value in params.items():
                if param_name in bounds:
                    if "min" in bounds[param_name] and param_value < bounds[param_name]["min"]:
                        raise ValueError(f"Parameter {param_name} below minimum: {param_value} < {bounds[param_name]['min']}")
                    if "max" in bounds[param_name] and param_value > bounds[param_name]["max"]:
                        raise ValueError(f"Parameter {param_name} above maximum: {param_value} > {bounds[param_name]['max']}")
            
            return InterventionSpec(
                lever_id=lever_id,
                parameters=params,
                rollback_descriptor={"lever_id": lever_id, "parameters": params} if lever_spec.rollback_required else None
            )
        
        return constructor
    
    def create_intervention(self, lever_id: str, **params) -> InterventionSpec:
        """
        Create an intervention from a lever ID and parameters.
        
        Args:
            lever_id: ID of the lever
            **params: Intervention parameters
            
        Returns:
            InterventionSpec: Intervention specification
        """
        if lever_id not in self.constructors:
            raise ValueError(f"Unknown lever ID: {lever_id}")
        return self.constructors[lever_id](**params)


class PolicyLoopMixin:
    """Mixin class for temporal policy loop execution.
    
    Provides methods for:
    - run_epoch(): Execute single epoch (observe → plan → act → update)
    - plan_actions(): Multi-objective constrained optimization
    - update_models(): Online learning (RLS + Bayesian regression)
    - explain_decision(): Generate rationale with decision hash
    """
    
    def __init__(
        self,
        doctrine: DoctrineV1,
        ledger: Ledger,
        seed: int = 42,
        sensor_registry: Optional[Any] = None,
        actuator_registry: Optional[Any] = None
    ):
        """
        Initialize policy loop mixin.
        
        Args:
            doctrine: Policy doctrine
            ledger: Event ledger
            seed: Random seed for determinism
            sensor_registry: Optional SensorRegistry instance
            actuator_registry: Optional ActuatorRegistry instance
        """
        self.doctrine = doctrine
        self.compiled_policy = CompiledPolicy.compile(doctrine)
        self.ledger = ledger
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Sensor and actuator registries
        self.sensor_registry = sensor_registry
        self.actuator_registry = actuator_registry
        
        # Auto-wire sensors/actuators from doctrine if registries provided
        if self.sensor_registry and SENSORS_AVAILABLE:
            # Auto-discover sensors
            self.sensor_registry.auto_discover()
        
        # MPC planner (optional)
        self.use_mpc = False  # Can be enabled via set_mpc_mode()
        self.mpc_planner: Optional[Any] = None
        if MPC_AVAILABLE and MPCPlanner:
            try:
                self.mpc_planner = MPCPlanner(doctrine, horizon=5)
            except Exception as e:
                logger.warning(f"Failed to initialize MPC planner: {e}")
        
        # Intervention catalog
        self.intervention_catalog = InterventionCatalog(doctrine)
        
        # Model state (learned parameters θ)
        self.model_state: Optional[ModelState] = None
        
        # Feature history for lag/EMA features
        self.feature_history: List[Dict[str, float]] = []
        self.max_history = 10  # Keep last 10 epochs
        
        # EMA parameters
        self.ema_alpha = 0.3  # EMA smoothing factor
        self.ema_features: Dict[str, float] = {}
        
        # Drift detection (CUSUM + optional ruptures)
        self.drift_detection_mode = "cusum"  # "cusum", "ruptures", "hybrid"
        self.cusum_stat = 0.0
        self.cusum_k = 0.5  # CUSUM threshold parameter
        self.cusum_h = 5.0  # CUSUM alarm threshold
        self.conservative_mode = False
        
        # Ruptures-based drift detection (optional)
        self.drift_detector: Optional[Any] = None
        self.metric_history: Dict[str, List[float]] = {}  # Per-metric history for ruptures
        if DRIFT_DETECTION_AVAILABLE and HybridDriftDetector:
            try:
                self.drift_detector = HybridDriftDetector(use_ruptures=True)
            except Exception as e:
                logger.warning(f"Failed to initialize drift detector: {e}")
        
        # Rollback manager (optional)
        self.rollback_manager: Optional[Any] = None
        self.auto_rollback_on_error = False  # Auto-rollback on actuator errors
        self.auto_rollback_on_invariant = False  # Auto-rollback on invariant violations
        
        # RLS parameters for transition model
        self.rls_lambda = 0.95  # Forgetting factor
        self.rls_A: Optional[np.ndarray] = None  # State transition matrix
        self.rls_B: Optional[np.ndarray] = None  # Action effect matrix
        self.rls_P: Dict[int, np.ndarray] = {}  # Parameter covariance (per output)
        self.rls_theta: Dict[int, np.ndarray] = {}  # Parameter vector (per output)
        
        # Bayesian regression for causal effects
        self.bayesian_tau = 1.0  # Prior variance
        self.bayesian_sigma = 0.1  # Observation noise
        self.bayesian_beta_mu: Optional[np.ndarray] = None  # Posterior mean
        self.bayesian_beta_sigma: Optional[np.ndarray] = None  # Posterior covariance
        
        # Action history for causal effect estimation
        self.action_history: List[Dict[str, Any]] = []
        self.metric_delta_history: List[Dict[str, float]] = []
    
    def extract_features(
        self,
        snapshot: Dict[str, float],
        k: int = 3
    ) -> np.ndarray:
        """
        Extract feature vector x_t = φ(s_t, L_{t-k:t}).
        
        Includes:
        - Instantaneous metrics
        - Lag features (x_{t-1}, x_{t-2}, ...)
        - EMA features
        - Volatility (windowed std)
        - Drift features (CUSUM/z-score)
        
        Args:
            snapshot: Current state snapshot (metric_name -> value)
            k: Number of lag features to include
            
        Returns:
            np.ndarray: Feature vector
        """
        # Get metric values in deterministic order
        metric_names = sorted(self.doctrine.metrics.keys())
        n_metrics = len(metric_names)
        
        # Instantaneous features
        instant_features = np.array([snapshot.get(name, 0.0) for name in metric_names])
        
        # Lag features
        lag_features = []
        for lag in range(1, min(k + 1, len(self.feature_history) + 1)):
            if len(self.feature_history) >= lag:
                lag_snapshot = self.feature_history[-lag]
                lag_vals = np.array([lag_snapshot.get(name, 0.0) for name in metric_names])
            else:
                lag_vals = np.zeros(n_metrics)
            lag_features.append(lag_vals)
        
        # Pad if needed
        while len(lag_features) < k:
            lag_features.insert(0, np.zeros(n_metrics))
        
        lag_features = np.concatenate(lag_features) if lag_features else np.array([])
        
        # EMA features
        ema_features = []
        for name in metric_names:
            current_val = snapshot.get(name, 0.0)
            if name in self.ema_features:
                ema_val = self.ema_alpha * current_val + (1 - self.ema_alpha) * self.ema_features[name]
            else:
                ema_val = current_val
            self.ema_features[name] = ema_val
            ema_features.append(ema_val)
        ema_features = np.array(ema_features)
        
        # Volatility (windowed std)
        if len(self.feature_history) >= 3:
            recent_values = [
                [hist.get(name, 0.0) for name in metric_names]
                for hist in self.feature_history[-3:]
            ]
            volatility = np.std(recent_values, axis=0)
        else:
            volatility = np.zeros(n_metrics)
        
        # Drift features (z-score of current vs EMA)
        drift_features = []
        for i, name in enumerate(metric_names):
            current_val = snapshot.get(name, 0.0)
            ema_val = self.ema_features.get(name, current_val)
            if volatility[i] > 1e-6:
                z_score = (current_val - ema_val) / volatility[i]
            else:
                z_score = 0.0
            drift_features.append(z_score)
        drift_features = np.array(drift_features)
        
        # Concatenate all features
        features = np.concatenate([
            instant_features,
            lag_features,
            ema_features,
            volatility,
            drift_features
        ])
        
        return features
    
    def update_transition_model(
        self,
        x_t: np.ndarray,
        u_t: np.ndarray,
        x_tp1: np.ndarray
    ) -> None:
        """
        Update transition model using Recursive Least Squares (RLS).
        
        Model: x_{t+1} = Ax_t + Bu_t + ε, ε ~ N(0, Σ)
        
        Args:
            x_t: State at time t
            u_t: Action vector at time t
            x_tp1: State at time t+1
        """
        # Construct regressor: Φ_t = [x_t; u_t]
        Phi_t = np.concatenate([x_t, u_t])
        n_inputs = len(Phi_t)
        n_outputs = len(x_tp1)
        
        # Initialize if needed - use per-output RLS
        if not self.rls_theta:
            # Will be initialized per output dimension below
            pass
        
        # For each output dimension, maintain separate RLS
        # We only learn to predict the instantaneous metrics (first n_metrics elements)
        n_metrics = len(self.doctrine.metrics)
        n_outputs_to_learn = min(n_outputs, n_metrics)
        
        for i in range(n_outputs_to_learn):
            # Check if we need to reinitialize due to size change
            if i not in self.rls_theta or len(self.rls_theta[i]) != n_inputs:
                # Initialize or reinitialize for this output dimension
                self.rls_theta[i] = np.zeros(n_inputs)
                self.rls_P[i] = np.eye(n_inputs) * 100.0  # Large initial covariance
            
            # Get regressor for this output
            phi = Phi_t
            
            # Kalman gain for this output
            P_phi = self.rls_P[i] @ phi
            denom = self.rls_lambda + np.dot(phi, P_phi)
            if denom < 1e-10:
                continue
            
            K = P_phi / denom
            
            # Prediction
            pred = np.dot(self.rls_theta[i], phi)
            # Target is the instantaneous metric value (first n_metrics elements of x_tp1)
            target_idx = i if i < len(x_tp1) else 0
            error = x_tp1[target_idx] - pred
            
            # Update parameter
            self.rls_theta[i] = self.rls_theta[i] + K * error
            
            # Update covariance
            self.rls_P[i] = (self.rls_P[i] - np.outer(K, P_phi)) / self.rls_lambda
        
        # Initialize A and B matrices if needed (for prediction)
        if self.rls_A is None:
            self.rls_A = np.eye(n_outputs) * 0.9  # Slight decay
            self.rls_B = np.zeros((n_outputs, max(1, n_inputs - n_outputs)))  # Action effects
        
        # Store updated model state
        theta_dict = {str(k): v.tolist() for k, v in self.rls_theta.items()} if isinstance(self.rls_theta, dict) else {}
        P_dict = {str(k): v.tolist() for k, v in self.rls_P.items()} if isinstance(self.rls_P, dict) else {}
        
        self.model_state = ModelState.create(
            parameters={
                "A": self.rls_A.tolist() if self.rls_A is not None else [],
                "B": self.rls_B.tolist() if self.rls_B is not None else [],
                "theta": theta_dict
            },
            covariance={
                "P": P_dict
            }
        )
    
    def predict_next_state(
        self,
        x_t: np.ndarray,
        u_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state and uncertainty.
        
        Args:
            x_t: Current state (feature vector)
            u_t: Action vector
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted_state, uncertainty)
        """
        Phi_t = np.concatenate([x_t, u_t])
        n_inputs = len(Phi_t)
        
        # Get number of metrics (output dimensions)
        n_metrics = len(self.doctrine.metrics)
        
        if not self.rls_theta:
            # No model yet - return current state with high uncertainty
            # Extract just the instantaneous metrics from feature vector
            instant_features = x_t[:n_metrics] if len(x_t) >= n_metrics else x_t
            return instant_features.copy(), np.ones(len(instant_features)) * 10.0
        
        # Predict using learned parameters
        # We predict the instantaneous metrics (first n_metrics elements of feature vector)
        x_pred = np.zeros(n_metrics)
        uncertainty = np.zeros(n_metrics)
        
        for i in range(n_metrics):
            if i in self.rls_theta:
                # Check if parameter vector size matches
                if len(self.rls_theta[i]) == n_inputs:
                    # Use learned parameters
                    x_pred[i] = np.dot(self.rls_theta[i], Phi_t)
                    # Uncertainty from covariance diagonal
                    if i in self.rls_P and self.rls_P[i].size > 0:
                        uncertainty[i] = np.sqrt(np.abs(self.rls_P[i][0, 0]))
                    else:
                        uncertainty[i] = 1.0
                else:
                    # Size mismatch - reinitialize for this output
                    self.rls_theta[i] = np.zeros(n_inputs)
                    self.rls_P[i] = np.eye(n_inputs) * 100.0
                    x_pred[i] = x_t[i] if i < len(x_t) else 0.0
                    uncertainty[i] = 10.0
            else:
                # Not initialized yet
                x_pred[i] = x_t[i] if i < len(x_t) else 0.0
                uncertainty[i] = 10.0
        
        return x_pred, uncertainty
    
    def update_causal_effects(self, epoch: int) -> None:
        """
        Update causal effect estimates using Bayesian linear regression.
        
        Model: Δm_t = β^T z_t + η_t
        
        Args:
            epoch: Current epoch
        """
        # Get deltas from ledger
        deltas = self.ledger.compute_deltas(epoch)
        if not deltas:
            return
        
        # Get recent actions and outcomes
        if len(self.action_history) == 0 or len(self.metric_delta_history) == 0:
            return
        
        # For each metric, estimate causal effect
        metric_names = sorted(self.doctrine.metrics.keys())
        
        for metric_name in metric_names:
            if metric_name not in deltas:
                continue
            
            # Collect data: (action_features, delta)
            Z = []  # Action features
            y = []  # Metric deltas
            
            for i, (action, delta_dict) in enumerate(zip(self.action_history, self.metric_delta_history)):
                if metric_name in delta_dict:
                    # Encode action as feature vector
                    action_vec = self._encode_action(action)
                    Z.append(action_vec)
                    y.append(delta_dict[metric_name])
            
            if len(Z) < 2:
                continue
            
            Z = np.array(Z)
            y = np.array(y)
            
            # Bayesian regression: β ~ N(0, τ²I), η ~ N(0, σ²)
            # Posterior: Σ_β = (σ⁻² Z^T Z + τ⁻² I)⁻¹, μ_β = σ⁻² Σ_β Z^T y
            
            n_features = Z.shape[1]
            I = np.eye(n_features)
            
            # Compute posterior
            ZTZ = Z.T @ Z
            sigma_inv_sq = 1.0 / (self.bayesian_sigma ** 2)
            tau_inv_sq = 1.0 / (self.bayesian_tau ** 2)
            
            try:
                if SCIPY_AVAILABLE:
                    Sigma_beta_inv = sigma_inv_sq * ZTZ + tau_inv_sq * I
                    Sigma_beta = scipy_linalg.inv(Sigma_beta_inv)
                else:
                    Sigma_beta_inv = sigma_inv_sq * ZTZ + tau_inv_sq * I
                    Sigma_beta = np.linalg.inv(Sigma_beta_inv)
                
                mu_beta = sigma_inv_sq * Sigma_beta @ Z.T @ y
                
                # Store for this metric
                if self.bayesian_beta_mu is None:
                    self.bayesian_beta_mu = {}
                if self.bayesian_beta_sigma is None:
                    self.bayesian_beta_sigma = {}
                
                self.bayesian_beta_mu[metric_name] = mu_beta
                self.bayesian_beta_sigma[metric_name] = Sigma_beta
                
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix in Bayesian regression for {metric_name}")
                continue
    
    def _encode_action(self, action: Dict[str, Any]) -> np.ndarray:
        """Encode action as feature vector for causal effect estimation."""
        # Simple encoding: one-hot for lever_id + parameter values
        lever_ids = sorted(self.doctrine.levers.keys())
        n_levers = len(lever_ids)
        
        # One-hot lever ID
        lever_vec = np.zeros(n_levers)
        if "lever_id" in action:
            try:
                idx = lever_ids.index(action["lever_id"])
                lever_vec[idx] = 1.0
            except ValueError:
                pass
        
        # Parameter values (normalized)
        param_vec = []
        if "parameters" in action:
            for lever_id in lever_ids:
                if lever_id in self.doctrine.levers:
                    lever_spec = self.doctrine.levers[lever_id]
                    for param_name, param_value in action["parameters"].items():
                        if param_name in lever_spec.bounds:
                            bounds = lever_spec.bounds[param_name]
                            if "min" in bounds and "max" in bounds:
                                # Normalize to [0, 1]
                                norm_val = (param_value - bounds["min"]) / (bounds["max"] - bounds["min"])
                                param_vec.append(norm_val)
                            else:
                                param_vec.append(param_value)
                        else:
                            param_vec.append(param_value)
        
        return np.concatenate([lever_vec, np.array(param_vec)])
    
    def set_mpc_mode(self, use_mpc: bool = True, horizon: int = 5, use_robust: bool = False) -> None:
        """
        Enable or disable MPC planning mode.
        
        Args:
            use_mpc: Whether to use MPC planner
            horizon: Prediction horizon for MPC
            use_robust: Whether to use robust MPC
        """
        self.use_mpc = use_mpc
        if use_mpc and MPC_AVAILABLE and MPCPlanner:
            try:
                self.mpc_planner = MPCPlanner(self.doctrine, horizon=horizon, use_robust=use_robust)
            except Exception as e:
                logger.warning(f"Failed to initialize MPC planner: {e}")
                self.use_mpc = False
    
    def plan_actions(
        self,
        x_t: np.ndarray,
        current_metrics: Dict[str, float],
        beam_width: int = 5
    ) -> Tuple[List[InterventionSpec], float, str]:
        """
        Plan actions using constrained multi-objective optimization.
        
        Uses MPC if enabled, otherwise falls back to beam search.
        
        Score: Σ w_i ΔĴ_i(a) - λ_c C(a) - λ_r R(a)
        
        Args:
            x_t: Current feature vector
            current_metrics: Current metric values
            beam_width: Beam search width (used if MPC not available)
            
        Returns:
            Tuple[List[InterventionSpec], float, str]: (chosen_interventions, score, rationale)
        """
        # Try MPC first if enabled
        if self.use_mpc and self.mpc_planner:
            try:
                # Prepare objectives and constraints
                objectives = [{"metric_name": obj.metric_name, "direction": obj.direction, "weight": self.compiled_policy.normalized_weights.get(obj.metric_name, 1.0)} for obj in self.doctrine.objectives]
                constraints = [{"condition": inv.condition} for inv in self.doctrine.invariants]
                lever_bounds = {lever_id: lever_spec.bounds for lever_id, lever_spec in self.doctrine.levers.items()}
                
                # Get transition model matrices
                A = self.rls_A
                B = self.rls_B
                
                interventions, score, rationale = self.mpc_planner.solve_mpc(
                    x_t, A, B, objectives, constraints, lever_bounds
                )
                
                if interventions:  # MPC succeeded
                    return interventions, score, f"MPC: {rationale}"
                else:
                    # Fall through to beam search
                    logger.debug(f"MPC returned no interventions, falling back to beam search: {rationale}")
            except Exception as e:
                logger.warning(f"MPC planning failed, falling back to beam search: {e}")
        
        # Fallback to beam search
        # Generate candidate actions from allowed levers
        candidates = []
        
        for lever_id, lever_spec in self.doctrine.levers.items():
            # Generate parameter combinations (simplified - grid search)
            bounds = lever_spec.bounds
            
            # Simple strategy: try a few parameter values
            param_combos = [{}]  # No-op action
            
            # Add some parameter variations
            for param_name, param_bounds in bounds.items():
                if "min" in param_bounds and "max" in param_bounds:
                    min_val = param_bounds["min"]
                    max_val = param_bounds["max"]
                    # Try low, medium, high
                    for val in [min_val, (min_val + max_val) / 2, max_val]:
                        new_combos = []
                        for combo in param_combos:
                            new_combo = combo.copy()
                            new_combo[param_name] = val
                            new_combos.append(new_combo)
                        param_combos.extend(new_combos)
            
            # Limit combinations
            param_combos = param_combos[:10]
            
            for params in param_combos:
                try:
                    intervention = self.intervention_catalog.create_intervention(lever_id, **params)
                    candidates.append(intervention)
                except Exception as e:
                    logger.debug(f"Invalid intervention: {e}")
                    continue
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # Encode action
            u_t = self._encode_action({"lever_id": candidate.lever_id, "parameters": candidate.parameters})
            
            # Predict effect
            x_pred, uncertainty = self.predict_next_state(x_t, u_t)
            
            # Compute objective improvement
            score = 0.0
            rationale_parts = []
            
            for obj in self.doctrine.objectives:
                metric_name = obj.metric_name
                weight = self.compiled_policy.normalized_weights.get(metric_name, 1.0)
                
                # Get predicted metric change (simplified - use first metric dimension)
                metric_idx = sorted(self.doctrine.metrics.keys()).index(metric_name) if metric_name in self.doctrine.metrics else 0
                if metric_idx < len(x_pred):
                    delta_pred = x_pred[metric_idx] - x_t[metric_idx] if metric_idx < len(x_t) else 0.0
                else:
                    delta_pred = 0.0
                
                # Use causal effect estimate if available
                if metric_name in (self.bayesian_beta_mu or {}):
                    beta_mu = self.bayesian_beta_mu[metric_name]
                    causal_effect = np.dot(beta_mu, u_t)
                    delta_pred = causal_effect
                
                # Objective contribution
                if obj.direction == "minimize":
                    improvement = -delta_pred * weight
                else:
                    improvement = delta_pred * weight
                
                score += improvement
                rationale_parts.append(f"{metric_name}: {improvement:.3f}")
            
            # Cost penalty
            cost = candidate.cost or 0.0
            score -= 0.1 * cost  # Cost penalty weight
            
            # Risk penalty (from uncertainty)
            risk = np.mean(uncertainty)
            score -= 0.2 * risk  # Risk penalty weight
            
            # Check invariants
            violates_invariant = False
            for inv in self.doctrine.invariants:
                # Simple invariant check (full version would evaluate condition)
                if "never_touch" in inv.condition.lower():
                    # Check if lever targets forbidden items
                    if any(forbidden in str(candidate.parameters) for forbidden in ["sshd", "NetworkManager"]):
                        violates_invariant = True
                        break
            
            if violates_invariant:
                score = -1e6  # Heavy penalty
            
            scored_candidates.append((candidate, score, " | ".join(rationale_parts)))
        
        # Beam search: keep top beam_width candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:beam_width]
        
        # Apply risk budget
        total_risk = 0.0
        selected = []
        
        for candidate, score, rationale in top_candidates:
            if len(selected) >= self.doctrine.risk_budget.max_actions_per_epoch:
                break
            
            # Estimate risk
            u_t = self._encode_action({"lever_id": candidate.lever_id, "parameters": candidate.parameters})
            _, uncertainty = self.predict_next_state(x_t, u_t)
            risk = np.mean(uncertainty)
            
            if total_risk + risk > self.doctrine.risk_budget.max_risk_per_epoch:
                if self.doctrine.risk_budget.rollback_required:
                    break  # Stop if rollback required
                else:
                    continue  # Skip this action
            
            selected.append(candidate)
            total_risk += risk
        
        # Return best action (or empty if conservative mode)
        if self.conservative_mode and len(selected) > 1:
            selected = selected[:1]  # Only one action in conservative mode
        
        if not selected:
            return [], 0.0, "No actions selected (risk budget or conservative mode)"
        
        best_score = top_candidates[0][1] if top_candidates else 0.0
        best_rationale = top_candidates[0][2] if top_candidates else "No rationale"
        
        return selected, best_score, best_rationale
    
    def set_drift_detection_mode(self, mode: str = "cusum") -> None:
        """
        Set drift detection mode.
        
        Args:
            mode: Detection mode ("cusum", "ruptures", "hybrid")
        """
        self.drift_detection_mode = mode
        if mode in ["ruptures", "hybrid"] and DRIFT_DETECTION_AVAILABLE and HybridDriftDetector:
            try:
                self.drift_detector = HybridDriftDetector(use_ruptures=(mode == "ruptures" or mode == "hybrid"))
            except Exception as e:
                logger.warning(f"Failed to initialize drift detector: {e}")
                self.drift_detection_mode = "cusum"
    
    def update_drift_detection(
        self,
        x_t: np.ndarray,
        x_pred: np.ndarray
    ) -> None:
        """
        Update drift detection using CUSUM or ruptures.
        
        Args:
            x_t: Actual state
            x_pred: Predicted state
        """
        n_metrics = len(self.doctrine.metrics)
        metric_names = sorted(self.doctrine.metrics.keys())
        
        # Update per-metric history for ruptures
        for i, metric_name in enumerate(metric_names):
            if i < len(x_t):
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                self.metric_history[metric_name].append(float(x_t[i]))
                # Keep only last 100 values
                if len(self.metric_history[metric_name]) > 100:
                    self.metric_history[metric_name].pop(0)
        
        # Use hybrid/ruptures if enabled
        if self.drift_detection_mode in ["ruptures", "hybrid"] and self.drift_detector:
            drift_detected = False
            for i, metric_name in enumerate(metric_names):
                if i < len(x_t) and i < len(x_pred):
                    detected, drift_info = self.drift_detector.update(
                        metric_name,
                        float(x_t[i]),
                        float(x_pred[i]),
                        self.metric_history.get(metric_name, [])
                    )
                    if detected:
                        drift_detected = True
                        logger.warning(f"Drift detected in {metric_name}: {drift_info}")
            
            # Update conservative mode
            if drift_detected:
                self.conservative_mode = True
            elif self.drift_detector.cusum_stat < self.cusum_h * 0.5:
                self.conservative_mode = False
            
            # Update CUSUM stat from detector
            self.cusum_stat = self.drift_detector.cusum_stat
        else:
            # Fallback to CUSUM
            residual = x_t - x_pred
            r_t = np.mean(np.abs(residual))  # Mean absolute residual
            
            # Update CUSUM
            self.cusum_stat = max(0.0, self.cusum_stat + r_t - self.cusum_k)
            
            # Check alarm
            if self.cusum_stat > self.cusum_h:
                if not self.conservative_mode:
                    logger.warning(f"Drift detected! CUSUM={self.cusum_stat:.3f} > {self.cusum_h}")
                    self.conservative_mode = True
            elif self.cusum_stat < self.cusum_h * 0.5:
                # Reset if well below threshold
                self.conservative_mode = False
    
    def explain_decision(
        self,
        interventions: List[InterventionSpec],
        score: float,
        rationale: str,
        epoch: int,
        x_t: np.ndarray
    ) -> Tuple[str, str]:
        """
        Generate decision explanation and compute decision hash.
        
        Args:
            interventions: Chosen interventions
            score: Optimization score
            rationale: Rationale text
            epoch: Current epoch
            x_t: Current feature vector
            
        Returns:
            Tuple[str, str]: (explanation, decision_hash)
        """
        # Build explanation
        explanation_parts = [
            f"Epoch {epoch}",
            f"Score: {score:.3f}",
            f"Rationale: {rationale}",
            f"Interventions: {len(interventions)}"
        ]
        
        for i, interv in enumerate(interventions):
            explanation_parts.append(f"  {i+1}. {interv.lever_id}: {interv.parameters}")
        
        explanation = "\n".join(explanation_parts)
        
        # Compute decision hash
        # Extract just the instantaneous metrics for state hash (deterministic)
        n_metrics = len(self.doctrine.metrics)
        x_t_metrics = x_t[:n_metrics] if len(x_t) >= n_metrics else x_t
        
        decision_data = {
            "epoch": epoch,
            "policy_hash": self.compiled_policy.policy_hash,
            "state_hash": stable_hash(x_t_metrics.tolist()),
            "interventions": [interv.model_dump() for interv in interventions],
            "score": score,
            "seed": self.seed
        }
        
        # Get ledger prefix hash (only observation/decision events, excluding timestamps)
        ledger_events = self.ledger.window(epoch, k=epoch + 1)
        # Hash only the payload content, not timestamps
        ledger_hashes = []
        for e in ledger_events:
            if e["type"] in ["observation", "decision"]:
                # Create deterministic hash from payload (excluding timestamp)
                payload = e["payload"].copy()
                if "timestamp" in payload:
                    del payload["timestamp"]
                ledger_hashes.append(stable_hash(payload))
        ledger_hash = stable_hash(sorted(ledger_hashes))  # Sort for determinism
        decision_data["ledger_prefix_hash"] = ledger_hash
        
        decision_hash = stable_hash(decision_data)
        
        return explanation, decision_hash
    
    def run_epoch(
        self,
        epoch: int,
        sensor_provider: Optional[Callable] = None,
        actuator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute single epoch: observe → plan → act → update.
        
        Args:
            epoch: Epoch number
            sensor_provider: Function to get current state snapshot (returns Dict[str, float])
            actuator: Function to execute interventions (takes List[InterventionSpec])
            
        Returns:
            Dict[str, Any]: Epoch results
        """
        # 1. Observe
        if sensor_provider:
            snapshot = sensor_provider()
        elif self.sensor_registry:
            # Use sensor registry if available
            required_metrics = list(self.doctrine.metrics.keys())
            snapshot = self.sensor_registry.read_all(required_metrics)
            # Fill missing metrics with 0.0
            for metric_name in required_metrics:
                if metric_name not in snapshot:
                    snapshot[metric_name] = 0.0
        else:
            # Dummy sensor for base implementation
            snapshot = {name: 0.0 for name in self.doctrine.metrics.keys()}
        
        # Extract features
        x_t = self.extract_features(snapshot)
        
        # Record observation
        obs_event = ObservationEvent(
            epoch=epoch,
            metrics=snapshot,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        ledger_event = LedgerEvent.from_observation(obs_event)
        self.ledger.append(ledger_event)
        
        # Update feature history
        self.feature_history.append(snapshot.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
        
        # 2. Plan
        interventions, score, rationale = self.plan_actions(x_t, snapshot)
        
        # Generate explanation and decision hash
        explanation, decision_hash = self.explain_decision(interventions, score, rationale, epoch, x_t)
        
        # Record decision
        decision_event = DecisionEvent(
            epoch=epoch,
            interventions=[interv.model_dump() for interv in interventions],
            rationale=rationale,
            decision_hash=decision_hash,
            score=score
        )
        ledger_event = LedgerEvent.from_decision(decision_event)
        self.ledger.append(ledger_event)
        
        # 3. Act
        checkpoint_id = None
        intervention_ids = []
        
        if interventions:
            # Create checkpoint before interventions if rollback enabled
            if self.rollback_manager:
                checkpoint_id = self.rollback_manager.create_checkpoint(epoch, snapshot)
            
            if actuator:
                # Use provided actuator function
                try:
                    actuator(interventions)
                    # Record interventions
                    if self.rollback_manager:
                        for interv in interventions:
                            interv_id = self.rollback_manager.record_intervention(
                                epoch, interv, result={"status": "success"}, checkpoint_id=checkpoint_id
                            )
                            intervention_ids.append(interv_id)
                except Exception as e:
                    logger.error(f"Actuator error: {e}")
                    # Record failed intervention
                    if self.rollback_manager:
                        for interv in interventions:
                            self.rollback_manager.record_intervention(
                                epoch, interv, result={"status": "error", "message": str(e)}, checkpoint_id=checkpoint_id
                            )
                    # Auto-rollback if enabled
                    if self.auto_rollback_on_error and self.rollback_manager:
                        logger.warning("Auto-rolling back due to actuator error")
                        self.rollback_manager.rollback(epoch, len(interventions), self.actuator_registry)
            elif self.actuator_registry:
                # Use actuator registry
                try:
                    result = self.actuator_registry.execute(interventions, transaction=True)
                    # Record interventions
                    if self.rollback_manager:
                        for i, interv in enumerate(interventions):
                            interv_result = result.get("results", [{}])[i] if i < len(result.get("results", [])) else {}
                            interv_id = self.rollback_manager.record_intervention(
                                epoch, interv, result=interv_result, checkpoint_id=checkpoint_id
                            )
                            intervention_ids.append(interv_id)
                    
                    if result.get("status") == "error":
                        logger.error(f"Actuator registry execution failed: {result.get('message')}")
                        # Auto-rollback if enabled
                        if self.auto_rollback_on_error and self.rollback_manager:
                            logger.warning("Auto-rolling back due to actuator registry error")
                            self.rollback_manager.rollback(epoch, len(interventions), self.actuator_registry)
                except Exception as e:
                    logger.error(f"Actuator registry error: {e}")
                    # Auto-rollback if enabled
                    if self.auto_rollback_on_error and self.rollback_manager:
                        logger.warning("Auto-rolling back due to actuator registry exception")
                        self.rollback_manager.rollback(epoch, len(interventions), self.actuator_registry)
        
        # Store action for causal effect estimation
        if interventions:
            for interv in interventions:
                self.action_history.append({
                    "lever_id": interv.lever_id,
                    "parameters": interv.parameters
                })
        
        # 4. Update (after observing outcome)
        # For base implementation, we'll simulate outcome
        # In real implementation, this would come from sensor after action
        if sensor_provider:
            outcome_snapshot = sensor_provider()  # Get outcome
        else:
            # Simulate outcome (simplified)
            outcome_snapshot = snapshot.copy()
            for interv in interventions:
                # Simple effect simulation
                for metric_name in outcome_snapshot:
                    outcome_snapshot[metric_name] += 0.1 * len(interventions)
        
        # Record outcome
        outcome_event = OutcomeEvent(
            epoch=epoch,
            metrics=outcome_snapshot,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        ledger_event = LedgerEvent.from_outcome(outcome_event)
        self.ledger.append(ledger_event)
        
        # Compute deltas
        deltas = {name: outcome_snapshot.get(name, 0.0) - snapshot.get(name, 0.0) for name in self.doctrine.metrics.keys()}
        self.metric_delta_history.append(deltas)
        if len(self.metric_delta_history) > self.max_history:
            self.metric_delta_history.pop(0)
        
        # Extract outcome features
        x_tp1 = self.extract_features(outcome_snapshot)
        
        # Encode action for transition model
        if interventions:
            u_t = self._encode_action({
                "lever_id": interventions[0].lever_id,
                "parameters": interventions[0].parameters
            })
        else:
            u_t = np.zeros(len(x_t))
        
        # Update transition model
        self.update_transition_model(x_t, u_t, x_tp1)
        
        # Update causal effects
        self.update_causal_effects(epoch)
        
        # Update drift detection
        # Extract just the instantaneous metrics for comparison
        n_metrics = len(self.doctrine.metrics)
        x_tp1_metrics = x_tp1[:n_metrics] if len(x_tp1) >= n_metrics else x_tp1
        x_pred_metrics, _ = self.predict_next_state(x_t, u_t)
        self.update_drift_detection(x_tp1_metrics, x_pred_metrics)
        
        return {
            "epoch": epoch,
            "observation": snapshot,
            "interventions": [interv.model_dump() for interv in interventions],
            "outcome": outcome_snapshot,
            "deltas": deltas,
            "score": score,
            "rationale": rationale,
            "decision_hash": decision_hash,
            "explanation": explanation,
            "conservative_mode": self.conservative_mode,
            "cusum_stat": self.cusum_stat,
            "checkpoint_id": checkpoint_id,
            "intervention_ids": intervention_ids
        }
    
    def rollback_last_n(self, epoch: int, n: int) -> List[str]:
        """
        Rollback last N interventions.
        
        Args:
            epoch: Current epoch
            n: Number of interventions to rollback
            
        Returns:
            List[str]: List of intervention IDs that were rolled back
        """
        if not self.rollback_manager:
            logger.warning("Rollback manager not initialized")
            return []
        
        return self.rollback_manager.rollback(epoch, n, self.actuator_registry)
    
    def enable_rollback(
        self,
        db_path: str = ":memory:",
        retention_days: int = 7,
        auto_rollback_on_error: bool = False,
        auto_rollback_on_invariant: bool = False
    ) -> None:
        """
        Enable rollback functionality.
        
        Args:
            db_path: Path to rollback database
            retention_days: Number of days to retain checkpoints
            auto_rollback_on_error: Auto-rollback on actuator errors
            auto_rollback_on_invariant: Auto-rollback on invariant violations
        """
        if ROLLBACK_AVAILABLE and RollbackManager:
            try:
                self.rollback_manager = RollbackManager(db_path, retention_days)
                self.auto_rollback_on_error = auto_rollback_on_error
                self.auto_rollback_on_invariant = auto_rollback_on_invariant
                logger.info("Rollback functionality enabled")
            except Exception as e:
                logger.error(f"Failed to enable rollback: {e}")
        else:
            logger.warning("Rollback not available (utils.rollback not importable)")

