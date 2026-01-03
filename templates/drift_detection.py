"""Drift detection using ruptures library for change-point detection.

Provides advanced change-point detection beyond simple CUSUM,
supporting multiple change points and per-metric detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

# Try to import ruptures
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning("ruptures not available - drift detection will use CUSUM only")


class DriftDetector:
    """Change-point detection using ruptures library.
    
    Supports multiple change-point detection algorithms:
    - Pelt: Optimal segmentation (penalized likelihood)
    - Dynp: Dynamic programming (exact solution)
    - Binseg: Binary segmentation (fast approximation)
    """
    
    def __init__(
        self,
        algorithm: str = "pelt",
        min_size: int = 2,
        penalty: float = 10.0,
        model: str = "rbf"
    ):
        """
        Initialize drift detector.
        
        Args:
            algorithm: Detection algorithm ("pelt", "dynp", "binseg")
            min_size: Minimum segment size
            penalty: Penalty parameter for Pelt
            model: Change-point model ("rbf", "l2", "l1", "normal", "ar")
        """
        self.algorithm = algorithm
        self.min_size = min_size
        self.penalty = penalty
        self.model = model
        self.detectors: Dict[str, Any] = {}  # Per-metric detectors
        self.change_points: Dict[str, List[int]] = {}  # Per-metric change points
        self.confidence_scores: Dict[str, List[float]] = {}  # Per-metric confidence
    
    def detect_changepoints(
        self,
        metric_name: str,
        metric_history: List[float],
        min_history: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Detect change points in metric history.
        
        Args:
            metric_name: Name of the metric
            metric_history: Historical values of the metric
            min_history: Minimum history length required
            
        Returns:
            Tuple of (change_point_indices, confidence_scores)
        """
        if not RUPTURES_AVAILABLE:
            return [], []
        
        if len(metric_history) < min_history:
            return [], []
        
        try:
            # Convert to numpy array
            signal = np.array(metric_history).reshape(-1, 1)
            
            # Initialize detector if needed
            if metric_name not in self.detectors:
                if self.algorithm == "pelt":
                    self.detectors[metric_name] = rpt.Pelt(model=self.model, min_size=self.min_size).fit(signal)
                elif self.algorithm == "dynp":
                    self.detectors[metric_name] = rpt.Dynp(model=self.model, min_size=self.min_size).fit(signal)
                elif self.algorithm == "binseg":
                    self.detectors[metric_name] = rpt.Binseg(model=self.model, min_size=self.min_size).fit(signal)
                else:
                    logger.warning(f"Unknown algorithm {self.algorithm}, using Pelt")
                    self.detectors[metric_name] = rpt.Pelt(model=self.model, min_size=self.min_size).fit(signal)
            else:
                # Update detector with new data
                self.detectors[metric_name].fit(signal)
            
            # Detect change points
            detector = self.detectors[metric_name]
            
            if self.algorithm == "pelt":
                change_points = detector.predict(pen=self.penalty)
            elif self.algorithm == "dynp":
                # For Dynp, need to specify number of change points
                # Use penalty to estimate number
                n_bkps = max(1, int(len(signal) / (self.penalty * 10)))
                change_points = detector.predict(n_bkps=n_bkps)
            elif self.algorithm == "binseg":
                n_bkps = max(1, int(len(signal) / (self.penalty * 10)))
                change_points = detector.predict(n_bkps=n_bkps)
            else:
                change_points = []
            
            # Remove last point (end of signal) if present
            if change_points and change_points[-1] == len(signal):
                change_points = change_points[:-1]
            
            # Compute confidence scores (simplified: based on segment variance differences)
            confidence_scores = []
            for cp in change_points:
                if cp > 0 and cp < len(signal):
                    # Compare variance before and after change point
                    before = signal[:cp]
                    after = signal[cp:]
                    var_before = np.var(before) if len(before) > 1 else 0.0
                    var_after = np.var(after) if len(after) > 1 else 0.0
                    # Confidence based on variance difference
                    confidence = min(1.0, abs(var_after - var_before) / (np.var(signal) + 1e-6))
                    confidence_scores.append(confidence)
                else:
                    confidence_scores.append(0.0)
            
            # Store results
            self.change_points[metric_name] = change_points
            self.confidence_scores[metric_name] = confidence_scores
            
            return change_points, confidence_scores
            
        except Exception as e:
            logger.error(f"Change-point detection failed for {metric_name}: {e}")
            return [], []
    
    def characterize_segment(
        self,
        metric_name: str,
        metric_history: List[float],
        change_points: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Characterize segments between change points.
        
        Args:
            metric_name: Name of the metric
            metric_history: Historical values
            change_points: List of change point indices
            
        Returns:
            List of segment characterizations
        """
        if not change_points:
            return []
        
        segments = []
        signal = np.array(metric_history)
        
        prev_cp = 0
        for cp in change_points + [len(signal)]:
            segment = signal[prev_cp:cp]
            if len(segment) > 0:
                trend_val = float(np.polyfit(range(len(segment)), segment, 1)[0]) if len(segment) > 1 else 0.0
                segments.append({
                    "start": prev_cp,
                    "end": cp,
                    "mean": float(np.mean(segment)),
                    "std": float(np.std(segment)),
                    "trend": trend_val,
                    "length": len(segment)
                })
            prev_cp = cp
        
        return segments
    
    def detect_drift(
        self,
        metric_name: str,
        metric_history: List[float],
        threshold: float = 0.7
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect if drift has occurred in a metric.
        
        Args:
            metric_name: Name of the metric
            metric_history: Historical values
            threshold: Confidence threshold for drift detection
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        change_points, confidence_scores = self.detect_changepoints(metric_name, metric_history)
        
        if not change_points:
            return False, None
        
        # Check if most recent change point has high confidence
        if confidence_scores:
            max_confidence = max(confidence_scores)
            if max_confidence >= threshold:
                # Get most recent change point
                recent_cp = change_points[-1]
                segments = self.characterize_segment(metric_name, metric_history, change_points)
                
                drift_info = {
                    "change_point": recent_cp,
                    "confidence": max_confidence,
                    "segments": segments,
                    "type": self._classify_drift_type(segments) if segments else "unknown"
                }
                
                return True, drift_info
        
        return False, None
    
    def _classify_drift_type(self, segments: List[Dict[str, Any]]) -> str:
        """Classify type of drift (mean shift, variance change, trend change)."""
        if len(segments) < 2:
            return "unknown"
        
        # Compare last two segments
        last_seg = segments[-1]
        prev_seg = segments[-2]
        
        mean_diff = abs(last_seg["mean"] - prev_seg["mean"])
        std_diff = abs(last_seg["std"] - prev_seg["std"])
        trend_diff = abs(last_seg["trend"] - prev_seg["trend"])
        
        if mean_diff > std_diff and mean_diff > trend_diff:
            return "mean_shift"
        elif std_diff > mean_diff and std_diff > trend_diff:
            return "variance_change"
        elif trend_diff > mean_diff and trend_diff > std_diff:
            return "trend_change"
        else:
            return "mixed"


class HybridDriftDetector:
    """Hybrid drift detector combining ruptures and CUSUM.
    
    Uses ruptures for change-point detection and CUSUM for continuous monitoring.
    """
    
    def __init__(
        self,
        use_ruptures: bool = True,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        ruptures_penalty: float = 10.0
    ):
        """
        Initialize hybrid detector.
        
        Args:
            use_ruptures: Whether to use ruptures for detection
            cusum_k: CUSUM threshold parameter
            cusum_h: CUSUM alarm threshold
            ruptures_penalty: Penalty for ruptures algorithm
        """
        self.use_ruptures = use_ruptures and RUPTURES_AVAILABLE
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.cusum_stat = 0.0
        
        if self.use_ruptures:
            self.ruptures_detector = DriftDetector(penalty=ruptures_penalty)
        else:
            self.ruptures_detector = None
    
    def update(
        self,
        metric_name: str,
        x_t: float,
        x_pred: float,
        metric_history: List[float]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Update drift detection with new observation.
        
        Args:
            metric_name: Name of the metric
            x_t: Actual value
            x_pred: Predicted value
            metric_history: Historical values
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        # CUSUM update
        residual = abs(x_t - x_pred)
        self.cusum_stat = max(0.0, self.cusum_stat + residual - self.cusum_k)
        
        cusum_alarm = self.cusum_stat > self.cusum_h
        
        # Ruptures detection (if enabled and enough history)
        ruptures_drift = False
        ruptures_info = None
        
        if self.use_ruptures and len(metric_history) >= 10:
            ruptures_drift, ruptures_info = self.ruptures_detector.detect_drift(
                metric_name, metric_history, threshold=0.7
            )
        
        # Combine results
        if ruptures_drift or cusum_alarm:
            drift_info = {
                "cusum_stat": self.cusum_stat,
                "cusum_alarm": cusum_alarm,
                "ruptures_drift": ruptures_drift,
                "ruptures_info": ruptures_info
            }
            return True, drift_info
        
        return False, None
    
    def reset(self) -> None:
        """Reset CUSUM statistic."""
        self.cusum_stat = 0.0

