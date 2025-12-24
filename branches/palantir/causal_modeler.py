"""
Causal Modeler for Vulnerability Chains

Integrates CRCA to model causal relationships between vulnerabilities,
network topology, and attack outcomes.
"""

from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from CRCA import CRCAAgent
    CRCA_AVAILABLE = True
except ImportError:
    CRCA_AVAILABLE = False
    logger.warning("CRCA not available, causal modeling disabled")

from palantir.device_discovery import Device
from palantir.network_mapper import NetworkGraph, RelationshipType


class CausalModeler:
    """
    Models causal relationships in network security using CRCA.
    
    Defines causal variables:
    - device_vulnerability_score: Vulnerability score of a device
    - network_exposure_level: Network exposure level
    - attack_success_probability: Probability of successful attack
    - lateral_movement_risk: Risk of lateral movement
    - data_exposure_risk: Risk of data exposure
    
    Args:
        crca_agent: Optional CRCA agent (creates new one if None)
    """
    
    def __init__(self, crca_agent: Optional[Any] = None) -> None:
        """
        Initialize causal modeler.
        
        Args:
            crca_agent: Optional CRCA agent instance.
        """
        if not CRCA_AVAILABLE:
            raise ImportError("CRCA is required for causal modeling")
        
        if crca_agent is None:
            # Create CRCA agent with security-focused variables
            self.crca_agent = CRCAAgent(
                variables=[
                    "device_vulnerability_score",
                    "network_exposure_level",
                    "attack_success_probability",
                    "lateral_movement_risk",
                    "data_exposure_risk",
                    "compromise_probability",
                    "cascade_failure_probability"
                ],
                enable_excel=False
            )
        else:
            self.crca_agent = crca_agent
        
        # Build causal relationships
        self._build_causal_model()
        
        logger.info("Causal modeler initialized")
    
    def _build_causal_model(self) -> None:
        """Build causal DAG for security relationships."""
        # Vulnerability score affects compromise probability
        self.crca_agent.add_causal_relationship(
            "device_vulnerability_score",
            "compromise_probability",
            strength=0.8,
            confidence=0.9
        )
        
        # Network exposure affects attack success
        self.crca_agent.add_causal_relationship(
            "network_exposure_level",
            "attack_success_probability",
            strength=0.7,
            confidence=0.85
        )
        
        # Compromise probability affects lateral movement
        self.crca_agent.add_causal_relationship(
            "compromise_probability",
            "lateral_movement_risk",
            strength=0.75,
            confidence=0.8
        )
        
        # Attack success affects data exposure
        self.crca_agent.add_causal_relationship(
            "attack_success_probability",
            "data_exposure_risk",
            strength=0.85,
            confidence=0.9
        )
        
        # Network exposure affects lateral movement
        self.crca_agent.add_causal_relationship(
            "network_exposure_level",
            "lateral_movement_risk",
            strength=0.6,
            confidence=0.75
        )
        
        # Lateral movement affects cascade failures
        self.crca_agent.add_causal_relationship(
            "lateral_movement_risk",
            "cascade_failure_probability",
            strength=0.7,
            confidence=0.8
        )
        
        logger.debug("Causal model built")
    
    def calculate_device_vulnerability_score(self, device: Device) -> float:
        """
        Calculate vulnerability score for a device.
        
        Args:
            device: Device to score
            
        Returns:
            Vulnerability score (0.0 to 1.0)
        """
        score = 0.0
        
        # Base score from number of vulnerabilities
        vuln_count = len(device.vulnerabilities)
        score += min(vuln_count * 0.2, 0.6)
        
        # No authentication adds significant risk
        if device.no_auth:
            score += 0.3
        
        # Known critical services add risk
        critical_services = {"ssh", "ftp", "telnet", "mysql", "postgresql", "redis"}
        if device.service.lower() in critical_services:
            score += 0.1
        
        return min(score, 1.0)
    
    def calculate_network_exposure_level(
        self,
        device: Device,
        network: NetworkGraph
    ) -> float:
        """
        Calculate network exposure level for a device.
        
        Args:
            device: Device to evaluate
            network: Network graph
            
        Returns:
            Exposure level (0.0 to 1.0)
        """
        exposure = 0.0
        
        # Count relationships (more connections = more exposure)
        neighbors = network.get_neighbors(device)
        exposure += min(len(neighbors) * 0.1, 0.5)
        
        # Same subnet increases exposure
        subnet_neighbors = [
            n for n in neighbors
            if any(
                rel == RelationshipType.SAME_SUBNET
                for src, tgt, rel in network.edges
                if (device == src and n == tgt) or (device == tgt and n == src)
            )
        ]
        if subnet_neighbors:
            exposure += 0.3
        
        # Public-facing services increase exposure
        public_ports = {80, 443, 8080, 8443}
        if device.port in public_ports:
            exposure += 0.2
        
        return min(exposure, 1.0)
    
    def build_device_state(
        self,
        device: Device,
        network: NetworkGraph
    ) -> Dict[str, float]:
        """
        Build state dictionary for CRCA from device and network.
        
        Args:
            device: Device to model
            network: Network graph
            
        Returns:
            State dictionary for CRCA
        """
        vulnerability_score = self.calculate_device_vulnerability_score(device)
        exposure_level = self.calculate_network_exposure_level(device, network)
        
        state = {
            "device_vulnerability_score": vulnerability_score,
            "network_exposure_level": exposure_level,
            "attack_success_probability": 0.0,  # Will be computed
            "lateral_movement_risk": 0.0,  # Will be computed
            "data_exposure_risk": 0.0,  # Will be computed
            "compromise_probability": 0.0,  # Will be computed
            "cascade_failure_probability": 0.0  # Will be computed
        }
        
        return state
    
    def predict_attack_outcomes(
        self,
        device: Device,
        network: NetworkGraph,
        interventions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Predict attack outcomes using CRCA.
        
        Args:
            device: Device to analyze
            network: Network graph
            interventions: Optional interventions to model
            
        Returns:
            Dictionary of predicted outcomes
        """
        initial_state = self.build_device_state(device, network)
        
        if interventions:
            # Apply interventions
            for key, value in interventions.items():
                if key in initial_state:
                    initial_state[key] = value
        
        # Run CRCA prediction
        result = self.crca_agent.run(
            initial_state=initial_state,
            max_steps=3
        )
        
        evolved_state = result.get("evolved_state", initial_state)
        
        return {
            "compromise_probability": evolved_state.get("compromise_probability", 0.0),
            "attack_success_probability": evolved_state.get("attack_success_probability", 0.0),
            "lateral_movement_risk": evolved_state.get("lateral_movement_risk", 0.0),
            "data_exposure_risk": evolved_state.get("data_exposure_risk", 0.0),
            "cascade_failure_probability": evolved_state.get("cascade_failure_probability", 0.0)
        }
    
    def model_vulnerability_chain(
        self,
        network: NetworkGraph,
        entry_device: Device
    ) -> Dict[str, Any]:
        """
        Model vulnerability chain starting from entry device.
        
        Args:
            network: Network graph
            entry_device: Entry point device
            
        Returns:
            Dictionary with chain analysis
        """
        logger.info(f"Modeling vulnerability chain from {entry_device.ip}")
        
        # Get reachable devices
        reachable = network.get_neighbors(entry_device)
        
        chain_analysis = {
            "entry_device": entry_device.ip,
            "reachable_devices": len(reachable),
            "device_risks": {},
            "chain_risk": 0.0
        }
        
        # Analyze each reachable device
        max_risk = 0.0
        for device in reachable:
            outcomes = self.predict_attack_outcomes(device, network)
            risk_score = (
                outcomes["compromise_probability"] * 0.4 +
                outcomes["lateral_movement_risk"] * 0.3 +
                outcomes["data_exposure_risk"] * 0.3
            )
            
            chain_analysis["device_risks"][device.ip] = {
                "risk_score": risk_score,
                **outcomes
            }
            
            max_risk = max(max_risk, risk_score)
        
        chain_analysis["chain_risk"] = max_risk
        
        return chain_analysis

