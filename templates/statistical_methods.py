"""
Statistical methods module for specialized agents.

Provides data-driven fitting, uncertainty quantification, time-series analysis,
Bayesian inference, sensitivity analysis, and attribution methods.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import math
import logging
import concurrent.futures

# Optional dependencies
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats  # type: ignore
    from scipy.optimize import minimize  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StatisticalMethods:
    """
    Statistical analysis methods for specialized agents.
    
    Provides methods for:
    - Data-driven model fitting
    - Uncertainty quantification
    - Time-series causality testing
    - Bayesian inference
    - Sensitivity analysis
    - Attribution methods
    """

    def __init__(
        self,
        graph_manager: Any,  # GraphManager instance
        prediction_framework: Any,  # PredictionFramework instance
        standardization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        edge_sign_constraints: Optional[Dict[Tuple[str, str], int]] = None,
        bayesian_priors: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
        seed: int = 42,
        bootstrap_workers: int = 0,
    ):
        """
        Initialize statistical methods.
        
        Args:
            graph_manager: GraphManager instance
            prediction_framework: PredictionFramework instance
            standardization_stats: Standardization statistics
            edge_sign_constraints: Edge sign constraints
            bayesian_priors: Bayesian priors for edges
            seed: Random seed
            bootstrap_workers: Number of workers for bootstrap
        """
        self.graph_manager = graph_manager
        self.prediction_framework = prediction_framework
        self.standardization_stats = standardization_stats or {}
        self.edge_sign_constraints = edge_sign_constraints or {}
        self.bayesian_priors = bayesian_priors or {}
        self.seed = seed
        self.bootstrap_workers = bootstrap_workers
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _require_pandas() -> None:
        """Raise ImportError if pandas is not available."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for this operation. Install pandas to proceed.")

    @staticmethod
    def _require_scipy() -> None:
        """Raise ImportError if scipy is not available."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for this operation. Install scipy to proceed.")

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
        
        Args:
            df: pandas DataFrame
            variables: List of variable names
            window: Rolling window size
            decay_alpha: Decay factor for recency weighting
            ridge_lambda: Ridge regularization parameter
            enforce_signs: Whether to enforce edge sign constraints
        """
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
        for node in self.graph_manager.get_nodes():
            if node not in self.standardization_stats:
                self.standardization_stats[node] = {"mean": 0.0, "std": 1.0}

        # Estimate edge strengths
        for child in self.graph_manager.get_nodes():
            parents = self.graph_manager.get_parents(child)
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
                self.graph_manager.add_relationship(p, child, strength=strength, confidence=1.0)

    def quantify_uncertainty(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95,
        agent_clone_factory: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty using bootstrap resampling.
        
        Args:
            df: pandas DataFrame
            variables: List of variable names
            windows: Number of bootstrap samples
            alpha: Confidence level
            agent_clone_factory: Optional factory function to create agent clones
            
        Returns:
            Dictionary with edge confidence intervals
        """
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"edge_cis": {}, "samples": 0}
        usable = df[variables].dropna()
        if len(usable) < 10:
            return {"edge_cis": {}, "samples": 0}
        windows = max(1, int(windows))
        samples: Dict[Tuple[str, str], List[float]] = {}

        # Snapshot current strengths
        baseline_strengths: Dict[Tuple[str, str], float] = {}
        for u, targets in self.graph_manager.graph.items():
            for v, meta in targets.items():
                try:
                    baseline_strengths[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                except Exception:
                    baseline_strengths[(u, v)] = 0.0
        baseline_stats = dict(self.standardization_stats)

        def _snapshot_strengths() -> Dict[Tuple[str, str], float]:
            snap: Dict[Tuple[str, str], float] = {}
            for u, targets in self.graph_manager.graph.items():
                for v, meta in targets.items():
                    try:
                        snap[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        snap[(u, v)] = 0.0
            return snap

        def _bootstrap_single(df_sample: "pd.DataFrame") -> Dict[Tuple[str, str], float]:
            if agent_clone_factory:
                clone = agent_clone_factory()
                clone.fit_from_dataframe(
                    df=df_sample,
                    variables=variables,
                    window=min(30, len(df_sample)),
                    decay_alpha=0.9,
                    ridge_lambda=0.0,
                    enforce_signs=True,
                )
                return _snapshot_strengths_from_graph(clone.graph_manager.graph)
            else:
                # Fallback: fit on current instance (not recommended for parallel)
                self.fit_from_dataframe(
                    df=df_sample,
                    variables=variables,
                    window=min(30, len(df_sample)),
                    decay_alpha=0.9,
                    ridge_lambda=0.0,
                    enforce_signs=True,
                )
                return _snapshot_strengths()

        def _snapshot_strengths_from_graph(graph: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
            res: Dict[Tuple[str, str], float] = {}
            for u, targets in graph.items():
                for v, meta in targets.items():
                    try:
                        res[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        res[(u, v)] = 0.0
            return res

        use_parallel = self.bootstrap_workers > 0 and agent_clone_factory is not None
        if use_parallel:
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
            if u in self.graph_manager.graph and v in self.graph_manager.graph[u]:
                self.graph_manager.graph[u][v]["strength"] = w
        self.standardization_stats = baseline_stats

        edge_cis: Dict[str, Tuple[float, float]] = {}
        for (u, v), arr in samples.items():
            arr_np = np.array(arr)
            lo = float(np.quantile(arr_np, (1 - alpha) / 2))
            hi = float(np.quantile(arr_np, 1 - (1 - alpha) / 2))
            edge_cis[f"{u}->{v}"] = (lo, hi)
        return {"edge_cis": edge_cis, "samples": windows}

    def granger_causality_test(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        """
        Perform Granger causality test.
        
        Args:
            df: pandas DataFrame
            var1: First variable (potential cause)
            var2: Second variable (potential effect)
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with test results
        """
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
        """
        Estimate vector autoregression model.
        
        Args:
            df: pandas DataFrame
            variables: List of variable names
            max_lag: Maximum lag
            
        Returns:
            Dictionary with VAR coefficients and residuals
        """
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
        Compute entropy and mutual information estimates.
        
        Args:
            df: pandas DataFrame
            variables: List of variable names
            
        Returns:
            Dictionary with entropies and mutual information
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

    def bayesian_edge_inference(
        self,
        df: Any,
        parent: str,
        child: str,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform Bayesian inference for edge strength.
        
        Args:
            df: pandas DataFrame
            parent: Parent variable
            child: Child variable
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
            
        Returns:
            Dictionary with posterior distribution parameters
        """
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
        """
        Perform sensitivity analysis.
        
        Args:
            intervention: Intervention dictionary
            target: Target variable
            perturbation_size: Size of perturbation
            
        Returns:
            Dictionary with sensitivities and elasticities
        """
        base_outcome = self.prediction_framework.predict_outcomes({}, intervention)
        base_target = base_outcome.get(target, 0.0)
        sensitivities: Dict[str, float] = {}
        elasticities: Dict[str, float] = {}
        for var, val in intervention.items():
            perturbed = dict(intervention)
            perturbed[var] = val + perturbation_size
            perturbed_outcome = self.prediction_framework.predict_outcomes({}, perturbed)
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
        """
        Perform deep root cause analysis.
        
        Args:
            problem_variable: Variable to analyze
            max_depth: Maximum search depth
            min_path_strength: Minimum path strength threshold
            
        Returns:
            Dictionary with root causes and paths
        """
        if problem_variable not in self.graph_manager.get_nodes():
            return {"error": f"Variable {problem_variable} not in graph"}
        all_ancestors = list(self.graph_manager.graph_reverse.get(problem_variable, []))
        root_causes: List[Dict[str, Any]] = []
        paths_to_problem: List[Dict[str, Any]] = []

        def path_strength(path: List[str]) -> float:
            prod = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                prod *= self.graph_manager.edge_strength(u, v)
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
                                "is_exogenous": len(self.graph_manager.get_parents(path[0])) == 0,
                            })
                            paths_to_problem.append({
                                "from": path[0],
                                "to": problem_variable,
                                "path": path,
                                "strength": float(ps),
                            })
                        continue
                    for child in self.graph_manager.get_children(node):
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
        """
        Compute Shapley value attribution.
        
        Args:
            baseline_state: Baseline state
            target_state: Target state
            target: Target variable
            samples: Number of samples
            
        Returns:
            Dictionary with Shapley values
        """
        variables = list(set(list(baseline_state.keys()) + list(target_state.keys())))
        n = len(variables)
        if n == 0:
            return {"shapley_values": {}, "normalized": {}, "total_attribution": 0.0}
        contributions: Dict[str, float] = {v: 0.0 for v in variables}

        def value(subset: List[str]) -> float:
            state = dict(baseline_state)
            for var in subset:
                if var in target_state:
                    state[var] = target_state[var]
            outcome = self.prediction_framework.predict_outcomes({}, state)
            return float(outcome.get(target, 0.0))

        for _ in range(max(1, samples)):
            perm = list(variables)
            self._rng.shuffle(perm)
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

    def gradient_based_intervention_optimization(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, Any]:
        """
        Optimize interventions using gradient-based optimization.
        
        Args:
            initial_state: Initial state
            target: Target variable
            intervention_vars: Variables that can be intervened on
            constraints: Optional bounds for interventions
            method: Optimization method
            
        Returns:
            Dictionary with optimal intervention and results
        """
        self._require_scipy()
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
            outcome = self.prediction_framework.predict_outcomes(initial_state, intervention)
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
            optimal_outcome = self.prediction_framework.predict_outcomes(initial_state, optimal_intervention)
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

