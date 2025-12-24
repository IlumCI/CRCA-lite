"""
Palantir Main Agent

Orchestrates all components to provide unified API for security intelligence.
"""

# Standard library imports
import os
from typing import Any, Dict, List, Optional

# Third-party imports
from loguru import logger

# Local imports
try:
    from CRCA import CRCAAgent

    CRCA_AVAILABLE = True
except ImportError:
    CRCA_AVAILABLE = False
    logger.warning("CRCA not available")

from palantir.attack_path_analyzer import (
    AttackPath,
    AttackPathAnalyzer,
    ThreatScenario,
)
from palantir.causal_modeler import CausalModeler
from palantir.device_discovery import Device, DeviceDiscovery
from palantir.network_mapper import NetworkGraph, NetworkMapper
from palantir.shodan_client import ShodanClient
from palantir.visualizer import Visualizer


class PalantirAgent:
    """
    Main Palantir agent orchestrating all components.
    
    Provides unified API for:
    - Device discovery via Shodan
    - Network topology mapping
    - Attack path analysis using CRCA
    - Threat scenario generation
    - Network visualization
    
    Args:
        shodan_api_key: Shodan API key (or from SHODAN_API_KEY env var)
        crca_agent: Optional CRCA agent (creates new one if None)
        prefer_no_auth: Prefer devices with no authentication (default: True)
    """
    
    def __init__(
        self,
        shodan_api_key: Optional[str] = None,
        crca_agent: Optional[Any] = None,
        prefer_no_auth: bool = True
    ) -> None:
        """
        Initialize Palantir agent.
        
        Args:
            shodan_api_key: Shodan API key. If None, reads from SHODAN_API_KEY env var.
            crca_agent: Optional CRCA agent instance.
            prefer_no_auth: Prefer devices with no authentication.
        """
        logger.info("Initializing Palantir agent...")
        
        # Initialize Shodan client
        self.shodan_client = ShodanClient(api_key=shodan_api_key)
        
        # Initialize device discovery
        self.device_discovery = DeviceDiscovery(prefer_no_auth=prefer_no_auth)
        
        # Initialize network mapper
        self.network_mapper = NetworkMapper()
        
        # Initialize causal modeler
        if CRCA_AVAILABLE:
            self.causal_modeler = CausalModeler(crca_agent=crca_agent)
        else:
            logger.warning("CRCA not available, causal modeling disabled")
            self.causal_modeler = None
        
        # Initialize attack path analyzer
        if self.causal_modeler:
            self.attack_path_analyzer = AttackPathAnalyzer(self.causal_modeler)
        else:
            self.attack_path_analyzer = None
        
        # Initialize visualizer
        try:
            self.visualizer = Visualizer()
        except ImportError:
            logger.warning("matplotlib not available, visualization disabled")
            self.visualizer = None
        
        logger.info("Palantir agent initialized")
    
    def discover_devices(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: Optional[int] = None,
        max_pages: int = 10
    ) -> List[Device]:
        """
        Discover devices using Shodan.
        
        Args:
            query: Shodan search query (e.g., "product:Apache httpd")
            filters: Optional filter criteria for devices
            max_results: Maximum number of results to return
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of discovered and filtered devices
        """
        logger.info(f"Discovering devices with query: {query}")
        
        # Search Shodan
        shodan_results = self.shodan_client.search_multiple_pages(
            query=query,
            max_results=max_results,
            max_pages=max_pages
        )
        
        # Discover and filter devices
        devices = self.device_discovery.discover_devices(
            shodan_results=shodan_results,
            filters=filters
        )
        
        logger.info(f"Discovered {len(devices)} devices")
        return devices
    
    def map_network(self, devices: List[Device]) -> NetworkGraph:
        """
        Map network topology from devices.
        
        Args:
            devices: List of devices to map
            
        Returns:
            NetworkGraph with devices and relationships
        """
        logger.info(f"Mapping network topology for {len(devices)} devices")
        
        network = self.network_mapper.map_network(devices)
        
        logger.info(
            f"Network mapped: {len(network.devices)} devices, "
            f"{len(network.edges)} relationships"
        )
        
        return network
    
    def analyze_attack_paths(
        self,
        network_graph: NetworkGraph,
        entry_point: Optional[Device] = None,
        target_device: Optional[Device] = None
    ) -> List[AttackPath]:
        """
        Analyze attack paths through the network.
        
        Args:
            network_graph: Network graph to analyze
            entry_point: Optional entry point device
            target_device: Optional target device
            
        Returns:
            List of attack paths
        """
        if not self.attack_path_analyzer:
            logger.error("Attack path analyzer not available (CRCA required)")
            return []
        
        if not entry_point:
            # Select highest-risk device as entry point
            entry_point = max(
                network_graph.devices,
                key=lambda d: len(d.vulnerabilities) + (1.0 if d.no_auth else 0.0)
            )
        
        logger.info(f"Analyzing attack paths from {entry_point.ip}")
        
        attack_paths = self.attack_path_analyzer.find_attack_paths(
            network=network_graph,
            entry_point=entry_point,
            target_device=target_device
        )
        
        logger.info(f"Found {len(attack_paths)} attack paths")
        return attack_paths
    
    def generate_threat_scenarios(
        self,
        network_graph: NetworkGraph,
        n_scenarios: int = 10,
        entry_point: Optional[Device] = None
    ) -> List[ThreatScenario]:
        """
        Generate threat scenarios using CRCA counterfactual analysis.
        
        Args:
            network_graph: Network graph
            n_scenarios: Number of scenarios to generate
            entry_point: Optional entry point device
            
        Returns:
            List of threat scenarios
        """
        if not self.attack_path_analyzer:
            logger.error("Attack path analyzer not available (CRCA required)")
            return []
        
        logger.info(f"Generating {n_scenarios} threat scenarios")
        
        scenarios = self.attack_path_analyzer.generate_threat_scenarios(
            network=network_graph,
            n_scenarios=n_scenarios,
            entry_point=entry_point
        )
        
        logger.info(f"Generated {len(scenarios)} threat scenarios")
        return scenarios
    
    def visualize_network(
        self,
        network_graph: NetworkGraph,
        attack_paths: Optional[List[AttackPath]] = None,
        output_file: Optional[str] = None,
        show_labels: bool = True,
        layout: str = "spring"
    ) -> None:
        """
        Visualize network topology and attack paths.
        
        Args:
            network_graph: Network graph to visualize
            attack_paths: Optional attack paths to highlight
            output_file: Optional output file path
            show_labels: Whether to show device labels
            layout: Layout algorithm
        """
        if not self.visualizer:
            logger.error("Visualizer not available (matplotlib required)")
            return
        
        self.visualizer.visualize_network(
            network=network_graph,
            attack_paths=attack_paths,
            output_file=output_file,
            show_labels=show_labels,
            layout=layout
        )
    
    def visualize_attack_paths(
        self,
        attack_paths: List[AttackPath],
        output_file: Optional[str] = None
    ) -> None:
        """
        Visualize attack paths.
        
        Args:
            attack_paths: List of attack paths
            output_file: Optional output file path
        """
        if not self.visualizer:
            logger.error("Visualizer not available (matplotlib required)")
            return
        
        self.visualizer.visualize_attack_paths(
            attack_paths=attack_paths,
            output_file=output_file
        )
    
    def visualize_vulnerability_heatmap(
        self,
        network_graph: NetworkGraph,
        output_file: Optional[str] = None
    ) -> None:
        """
        Create vulnerability heatmap.
        
        Args:
            network_graph: Network graph
            output_file: Optional output file path
        """
        if not self.visualizer:
            logger.error("Visualizer not available (matplotlib required)")
            return
        
        self.visualizer.visualize_vulnerability_heatmap(
            network=network_graph,
            output_file=output_file
        )
    
    def analyze_critical_paths(
        self,
        network_graph: NetworkGraph
    ) -> Dict[str, Any]:
        """
        Analyze critical attack paths in the network.
        
        Args:
            network_graph: Network graph
            
        Returns:
            Dictionary with critical path analysis
        """
        if not self.attack_path_analyzer:
            logger.error("Attack path analyzer not available (CRCA required)")
            return {}
        
        return self.attack_path_analyzer.analyze_critical_paths(network_graph)
    
    def full_analysis(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: Optional[int] = 100,
        n_scenarios: int = 10,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform full analysis pipeline: discover → map → analyze → visualize.
        
        Args:
            query: Shodan search query
            filters: Optional device filters
            max_results: Maximum devices to discover
            n_scenarios: Number of threat scenarios to generate
            output_dir: Optional output directory for visualizations
            
        Returns:
            Dictionary with full analysis results
        """
        logger.info("Starting full analysis pipeline")
        
        # Discover devices
        devices = self.discover_devices(
            query=query,
            filters=filters,
            max_results=max_results
        )
        
        if not devices:
            logger.warning("No devices discovered")
            return {"error": "No devices discovered"}
        
        # Map network
        network = self.map_network(devices)
        
        # Analyze attack paths
        attack_paths = self.analyze_attack_paths(network_graph=network)
        
        # Generate threat scenarios
        scenarios = self.generate_threat_scenarios(
            network_graph=network,
            n_scenarios=n_scenarios
        )
        
        # Analyze critical paths
        critical_paths = self.analyze_critical_paths(network_graph=network)
        
        # Visualize
        if self.visualizer and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            self.visualize_network(
                network_graph=network,
                attack_paths=attack_paths[:5],
                output_file=f"{output_dir}/network_topology.png"
            )
            
            if attack_paths:
                self.visualize_attack_paths(
                    attack_paths=attack_paths[:10],
                    output_file=f"{output_dir}/attack_paths.png"
                )
            
            self.visualize_vulnerability_heatmap(
                network_graph=network,
                output_file=f"{output_dir}/vulnerability_heatmap.png"
            )
        
        results = {
            "devices_discovered": len(devices),
            "network_size": len(network.devices),
            "relationships": len(network.edges),
            "communities": len(network.communities),
            "attack_paths": len(attack_paths),
            "threat_scenarios": len(scenarios),
            "critical_paths": critical_paths,
            "top_attack_paths": [
                {
                    "path": [d.ip for d in p.path],
                    "risk_score": p.risk_score,
                    "probability": p.probability
                }
                for p in attack_paths[:5]
            ],
            "top_scenarios": [
                {
                    "name": s.name,
                    "entry": s.entry_point.ip,
                    "target": s.target_device.ip,
                    "probability": s.probability
                }
                for s in scenarios[:5]
            ]
        }
        
        logger.info("Full analysis completed")
        return results

