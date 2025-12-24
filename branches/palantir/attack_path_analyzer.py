"""
Attack Path Analyzer

Uses CRCA counterfactual analysis to generate attack scenarios and identify
critical attack paths through the network.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from palantir.device_discovery import Device
from palantir.network_mapper import NetworkGraph
from palantir.causal_modeler import CausalModeler


@dataclass
class AttackPath:
    """
    Represents an attack path through the network.
    
    Args:
        path: Sequence of devices in the attack path
        risk_score: Overall risk score for the path
        probability: Probability of successful attack along this path
        reasoning: Explanation of why this path is dangerous
    """
    
    path: List[Device]
    risk_score: float
    probability: float
    reasoning: str


@dataclass
class ThreatScenario:
    """
    Represents a threat scenario generated from counterfactual analysis.
    
    Args:
        name: Scenario name
        entry_point: Entry point device
        target_device: Target device
        attack_path: Attack path to target
        interventions: Interventions that enable the attack
        expected_outcomes: Expected outcomes if attack succeeds
        probability: Probability of scenario
        reasoning: Explanation of the scenario
    """
    
    name: str
    entry_point: Device
    target_device: Device
    attack_path: List[Device]
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float
    reasoning: str


class AttackPathAnalyzer:
    """
    Analyzes attack paths using CRCA counterfactual reasoning.
    
    Args:
        causal_modeler: Causal modeler instance
    """
    
    def __init__(self, causal_modeler: CausalModeler) -> None:
        """
        Initialize attack path analyzer.
        
        Args:
            causal_modeler: Causal modeler instance
        """
        self.causal_modeler = causal_modeler
        logger.info("Attack path analyzer initialized")
    
    def find_attack_paths(
        self,
        network: NetworkGraph,
        entry_point: Device,
        target_device: Optional[Device] = None
    ) -> List[AttackPath]:
        """
        Find attack paths from entry point to target.
        
        Args:
            network: Network graph
            entry_point: Entry point device
            target_device: Optional target device (if None, finds all paths)
            
        Returns:
            List of attack paths
        """
        logger.info(f"Finding attack paths from {entry_point.ip}")
        
        if not network.graph:
            # Fallback: simple path finding
            return self._find_simple_paths(network, entry_point, target_device)
        
        try:
            import networkx as nx
            
            paths: List[AttackPath] = []
            
            if target_device:
                # Find all simple paths to target
                try:
                    all_paths = list(
                        nx.all_simple_paths(
                            network.graph,
                            entry_point,
                            target_device,
                            cutoff=5  # Limit path length
                        )
                    )
                except nx.NetworkXNoPath:
                    all_paths = []
            else:
                # Find paths to all reachable devices
                reachable = nx.node_connected_component(network.graph, entry_point)
                all_paths = []
                for target in reachable:
                    if target != entry_point:
                        try:
                            paths_to_target = list(
                                nx.all_simple_paths(
                                    network.graph,
                                    entry_point,
                                    target,
                                    cutoff=5
                                )
                            )
                            all_paths.extend(paths_to_target)
                        except nx.NetworkXNoPath:
                            continue
            
            # Score each path
            for path_nodes in all_paths:
                path = list(path_nodes)
                risk_score, probability, reasoning = self._score_path(path, network)
                
                attack_path = AttackPath(
                    path=path,
                    risk_score=risk_score,
                    probability=probability,
                    reasoning=reasoning
                )
                paths.append(attack_path)
            
            # Sort by risk score
            paths.sort(key=lambda p: p.risk_score, reverse=True)
            
            logger.info(f"Found {len(paths)} attack paths")
            return paths
            
        except Exception as e:
            logger.error(f"Error finding attack paths: {e}")
            return self._find_simple_paths(network, entry_point, target_device)
    
    def _find_simple_paths(
        self,
        network: NetworkGraph,
        entry_point: Device,
        target_device: Optional[Device]
    ) -> List[AttackPath]:
        """Fallback simple path finding without NetworkX."""
        paths: List[AttackPath] = []
        
        if target_device:
            # Simple BFS to find path
            visited = {entry_point}
            queue = [(entry_point, [entry_point])]
            
            while queue:
                current, path = queue.pop(0)
                
                if current == target_device:
                    risk_score, probability, reasoning = self._score_path(path, network)
                    paths.append(AttackPath(
                        path=path,
                        risk_score=risk_score,
                        probability=probability,
                        reasoning=reasoning
                    ))
                    break
                
                neighbors = network.get_neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        else:
            # Find paths to all neighbors
            neighbors = network.get_neighbors(entry_point)
            for neighbor in neighbors:
                path = [entry_point, neighbor]
                risk_score, probability, reasoning = self._score_path(path, network)
                paths.append(AttackPath(
                    path=path,
                    risk_score=risk_score,
                    probability=probability,
                    reasoning=reasoning
                ))
        
        paths.sort(key=lambda p: p.risk_score, reverse=True)
        return paths
    
    def _score_path(
        self,
        path: List[Device],
        network: NetworkGraph
    ) -> Tuple[float, float, str]:
        """
        Score an attack path.
        
        Args:
            path: Sequence of devices in path
            network: Network graph
            
        Returns:
            Tuple of (risk_score, probability, reasoning)
        """
        if not path:
            return 0.0, 0.0, "Empty path"
        
        # Calculate cumulative risk
        total_risk = 0.0
        total_probability = 1.0
        risk_factors: List[str] = []
        
        for device in path:
            outcomes = self.causal_modeler.predict_attack_outcomes(device, network)
            
            device_risk = (
                outcomes["compromise_probability"] * 0.4 +
                outcomes["lateral_movement_risk"] * 0.3 +
                outcomes["data_exposure_risk"] * 0.3
            )
            
            total_risk += device_risk
            total_probability *= outcomes["compromise_probability"]
            
            if device.no_auth:
                risk_factors.append(f"{device.ip} has no authentication")
            if device.vulnerabilities:
                risk_factors.append(
                    f"{device.ip} has {len(device.vulnerabilities)} vulnerabilities"
                )
        
        # Average risk across path
        avg_risk = total_risk / len(path) if path else 0.0
        
        # Path length penalty (longer paths are less likely)
        length_penalty = 0.9 ** (len(path) - 1)
        final_probability = total_probability * length_penalty
        
        reasoning = f"Path through {len(path)} devices. " + "; ".join(risk_factors[:3])
        
        return avg_risk, final_probability, reasoning
    
    def generate_threat_scenarios(
        self,
        network: NetworkGraph,
        n_scenarios: int = 10,
        entry_point: Optional[Device] = None
    ) -> List[ThreatScenario]:
        """
        Generate threat scenarios using CRCA counterfactual analysis.
        
        Args:
            network: Network graph
            n_scenarios: Number of scenarios to generate
            entry_point: Optional entry point device
            
        Returns:
            List of threat scenarios
        """
        logger.info(f"Generating {n_scenarios} threat scenarios")
        
        if not entry_point:
            # Select highest-risk device as entry point
            entry_point = max(
                network.devices,
                key=lambda d: len(d.vulnerabilities) + (1.0 if d.no_auth else 0.0)
            )
        
        # Build initial state for entry point
        initial_state = self.causal_modeler.build_device_state(entry_point, network)
        
        # Generate counterfactual scenarios using CRCA
        result = self.causal_modeler.crca_agent.run(
            initial_state=initial_state,
            target_variables=["lateral_movement_risk", "data_exposure_risk"],
            max_steps=3
        )
        
        scenarios: List[ThreatScenario] = []
        counterfactuals = result.get("counterfactual_scenarios", [])
        
        # Convert counterfactuals to threat scenarios
        for i, cf in enumerate(counterfactuals[:n_scenarios]):
            # Find target device (highest risk reachable device)
            reachable = network.get_neighbors(entry_point)
            if not reachable:
                continue
            
            target = max(
                reachable,
                key=lambda d: self.causal_modeler.calculate_device_vulnerability_score(d)
            )
            
            # Find attack path
            paths = self.find_attack_paths(network, entry_point, target)
            attack_path = paths[0].path if paths else [entry_point, target]
            
            scenario = ThreatScenario(
                name=f"Scenario {i+1}: {entry_point.ip} → {target.ip}",
                entry_point=entry_point,
                target_device=target,
                attack_path=attack_path,
                interventions=getattr(cf, "interventions", {}),
                expected_outcomes=getattr(cf, "expected_outcomes", {}),
                probability=getattr(cf, "probability", 0.5),
                reasoning=getattr(cf, "reasoning", "Generated from counterfactual analysis")
            )
            scenarios.append(scenario)
        
        # If not enough scenarios from counterfactuals, generate additional ones
        if len(scenarios) < n_scenarios:
            scenarios.extend(
                self._generate_additional_scenarios(
                    network,
                    entry_point,
                    n_scenarios - len(scenarios)
                )
            )
        
        logger.info(f"Generated {len(scenarios)} threat scenarios")
        return scenarios
    
    def _generate_additional_scenarios(
        self,
        network: NetworkGraph,
        entry_point: Device,
        count: int
    ) -> List[ThreatScenario]:
        """Generate additional scenarios using path analysis."""
        scenarios: List[ThreatScenario] = []
        
        # Find attack paths
        paths = self.find_attack_paths(network, entry_point)
        
        for i, path_obj in enumerate(paths[:count]):
            if len(path_obj.path) < 2:
                continue
            
            target = path_obj.path[-1]
            
            scenario = ThreatScenario(
                name=f"Path Scenario {i+1}: {entry_point.ip} → {target.ip}",
                entry_point=entry_point,
                target_device=target,
                attack_path=path_obj.path,
                interventions={},
                expected_outcomes={
                    "compromise_probability": path_obj.probability,
                    "risk_score": path_obj.risk_score
                },
                probability=path_obj.probability,
                reasoning=path_obj.reasoning
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def analyze_critical_paths(
        self,
        network: NetworkGraph
    ) -> Dict[str, Any]:
        """
        Analyze critical attack paths in the network.
        
        Args:
            network: Network graph
            
        Returns:
            Dictionary with critical path analysis
        """
        logger.info("Analyzing critical attack paths")
        
        # Find highest-risk entry points
        entry_points = sorted(
            network.devices,
            key=lambda d: (
                len(d.vulnerabilities) +
                (1.0 if d.no_auth else 0.0) +
                len(network.get_neighbors(d)) * 0.1
            ),
            reverse=True
        )[:5]
        
        critical_paths: List[AttackPath] = []
        
        for entry_point in entry_points:
            paths = self.find_attack_paths(network, entry_point)
            critical_paths.extend(paths[:3])  # Top 3 paths per entry point
        
        # Sort all paths by risk
        critical_paths.sort(key=lambda p: p.risk_score, reverse=True)
        
        return {
            "critical_paths": critical_paths[:10],  # Top 10 overall
            "entry_points": [ep.ip for ep in entry_points],
            "total_paths_analyzed": len(critical_paths)
        }

