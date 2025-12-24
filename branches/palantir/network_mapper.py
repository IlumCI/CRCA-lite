"""
Network Topology Mapper

Maps relationships between devices to build network graphs.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from loguru import logger

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available, graph features limited")

from palantir.device_discovery import Device


class RelationshipType(Enum):
    """Types of relationships between devices."""
    
    SAME_SUBNET = "same_subnet"
    SAME_ASN = "same_asn"
    SAME_ORG = "same_org"
    SAME_SERVICE = "same_service"
    SAME_VULNERABILITY = "same_vulnerability"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    SERVICE_DEPENDENCY = "service_dependency"


@dataclass
class NetworkGraph:
    """
    Network graph representation of devices and relationships.
    
    Args:
        devices: List of devices in the network
        edges: List of edges (source_device, target_device, relationship_type)
        communities: List of device communities/clusters
        graph: NetworkX graph object (if available)
    """
    
    devices: List[Device] = field(default_factory=list)
    edges: List[Tuple[Device, Device, RelationshipType]] = field(default_factory=list)
    communities: List[List[Device]] = field(default_factory=list)
    graph: Optional[Any] = None
    
    def __post_init__(self) -> None:
        """Initialize NetworkX graph if available."""
        if NETWORKX_AVAILABLE and self.graph is None:
            self.graph = nx.Graph()
            self._build_networkx_graph()
    
    def _build_networkx_graph(self) -> None:
        """Build NetworkX graph from devices and edges."""
        if not NETWORKX_AVAILABLE:
            return
        
        # Add nodes
        for device in self.devices:
            self.graph.add_node(
                device,
                ip=device.ip,
                port=device.port,
                service=device.service,
                device_type=device.device_type.value,
                vulnerabilities=len(device.vulnerabilities),
                no_auth=device.no_auth
            )
        
        # Add edges
        for source, target, rel_type in self.edges:
            if source in self.devices and target in self.devices:
                self.graph.add_edge(
                    source,
                    target,
                    relationship=rel_type.value,
                    weight=1.0
                )
    
    def get_device_by_ip(self, ip: str) -> Optional[Device]:
        """Get device by IP address."""
        for device in self.devices:
            if device.ip == ip:
                return device
        return None
    
    def get_neighbors(self, device: Device) -> List[Device]:
        """Get neighboring devices."""
        neighbors: Set[Device] = set()
        for source, target, _ in self.edges:
            if source == device:
                neighbors.add(target)
            elif target == device:
                neighbors.add(source)
        return list(neighbors)
    
    def get_community(self, device: Device) -> Optional[List[Device]]:
        """Get community/cluster containing device."""
        for community in self.communities:
            if device in community:
                return community
        return None


class NetworkMapper:
    """
    Maps network topology and relationships between devices.
    
    Args:
        subnet_threshold: Subnet mask for same-subnet detection (default: /24)
        geo_proximity_km: Geographic proximity threshold in km (default: 100)
    """
    
    def __init__(
        self,
        subnet_threshold: int = 24,
        geo_proximity_km: float = 100.0
    ) -> None:
        """
        Initialize network mapper.
        
        Args:
            subnet_threshold: Subnet mask for same-subnet detection.
            geo_proximity_km: Geographic proximity threshold in kilometers.
        """
        self.subnet_threshold = subnet_threshold
        self.geo_proximity_km = geo_proximity_km
        logger.info(
            f"Network mapper initialized "
            f"(subnet_threshold=/{subnet_threshold}, geo_proximity={geo_proximity_km}km)"
        )
    
    def _same_subnet(self, device1: Device, device2: Device) -> bool:
        """Check if two devices are on the same subnet."""
        try:
            ip1 = ipaddress.ip_address(device1.ip)
            ip2 = ipaddress.ip_address(device2.ip)
            
            # Get network for each IP with threshold mask
            net1 = ipaddress.ip_network(f"{ip1}/{self.subnet_threshold}", strict=False)
            net2 = ipaddress.ip_network(f"{ip2}/{self.subnet_threshold}", strict=False)
            
            return net1 == net2
        except ValueError:
            return False
    
    def _same_asn(self, device1: Device, device2: Device) -> bool:
        """Check if two devices share the same ASN."""
        return (
            device1.asn is not None
            and device2.asn is not None
            and device1.asn == device2.asn
        )
    
    def _same_org(self, device1: Device, device2: Device) -> bool:
        """Check if two devices belong to the same organization."""
        return (
            device1.org is not None
            and device2.org is not None
            and device1.org.lower() == device2.org.lower()
        )
    
    def _same_service(self, device1: Device, device2: Device) -> bool:
        """Check if two devices run the same service."""
        return device1.service.lower() == device2.service.lower()
    
    def _same_vulnerability(self, device1: Device, device2: Device) -> bool:
        """Check if two devices share any vulnerability."""
        vulns1 = set(device1.vulnerabilities)
        vulns2 = set(device2.vulnerabilities)
        return bool(vulns1 & vulns2)
    
    def _geographic_proximity(
        self,
        device1: Device,
        device2: Device
    ) -> bool:
        """
        Check if two devices are geographically proximate.
        
        Uses Haversine formula if location data available.
        """
        if not device1.location or not device2.location:
            return False
        
        loc1 = device1.location
        loc2 = device2.location
        
        lat1 = loc1.get("latitude")
        lon1 = loc1.get("longitude")
        lat2 = loc2.get("latitude")
        lon2 = loc2.get("longitude")
        
        if not all([lat1, lon1, lat2, lon2]):
            return False
        
        # Haversine formula
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371.0  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        
        return distance <= self.geo_proximity_km
    
    def _detect_relationships(
        self,
        device1: Device,
        device2: Device
    ) -> List[RelationshipType]:
        """
        Detect all relationships between two devices.
        
        Args:
            device1: First device
            device2: Second device
            
        Returns:
            List of relationship types
        """
        relationships: List[RelationshipType] = []
        
        if self._same_subnet(device1, device2):
            relationships.append(RelationshipType.SAME_SUBNET)
        
        if self._same_asn(device1, device2):
            relationships.append(RelationshipType.SAME_ASN)
        
        if self._same_org(device1, device2):
            relationships.append(RelationshipType.SAME_ORG)
        
        if self._same_service(device1, device2):
            relationships.append(RelationshipType.SAME_SERVICE)
        
        if self._same_vulnerability(device1, device2):
            relationships.append(RelationshipType.SAME_VULNERABILITY)
        
        if self._geographic_proximity(device1, device2):
            relationships.append(RelationshipType.GEOGRAPHIC_PROXIMITY)
        
        return relationships
    
    def map_network(self, devices: List[Device]) -> NetworkGraph:
        """
        Map network topology from list of devices.
        
        Args:
            devices: List of devices to map
            
        Returns:
            NetworkGraph with devices and relationships
        """
        logger.info(f"Mapping network topology for {len(devices)} devices")
        
        graph = NetworkGraph(devices=devices.copy())
        edges: List[Tuple[Device, Device, RelationshipType]] = []
        
        # Detect relationships between all device pairs
        for i, device1 in enumerate(devices):
            for device2 in devices[i + 1:]:
                relationships = self._detect_relationships(device1, device2)
                
                for rel_type in relationships:
                    edges.append((device1, device2, rel_type))
        
        graph.edges = edges
        logger.info(f"Found {len(edges)} relationships")
        
        # Detect communities using NetworkX if available
        if NETWORKX_AVAILABLE:
            graph._build_networkx_graph()
            if graph.graph:
                try:
                    communities = nx.community.greedy_modularity_communities(graph.graph)
                    graph.communities = [list(community) for community in communities]
                    logger.info(f"Detected {len(graph.communities)} communities")
                except Exception as e:
                    logger.warning(f"Could not detect communities: {e}")
        
        return graph
    
    def find_attack_surface(
        self,
        network: NetworkGraph,
        entry_point: Device
    ) -> List[Device]:
        """
        Find attack surface reachable from entry point.
        
        Uses graph traversal to find all devices reachable from entry point.
        
        Args:
            network: Network graph
            entry_point: Entry point device
            
        Returns:
            List of reachable devices
        """
        if not NETWORKX_AVAILABLE or not network.graph:
            # Fallback: return neighbors
            return network.get_neighbors(entry_point)
        
        try:
            # Find all nodes reachable from entry point
            reachable = nx.node_connected_component(network.graph, entry_point)
            return list(reachable)
        except Exception as e:
            logger.warning(f"Error finding attack surface: {e}")
            return network.get_neighbors(entry_point)

