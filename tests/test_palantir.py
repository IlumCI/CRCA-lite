"""
Unit tests for Palantir module.

Tests core components without requiring Shodan API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from palantir.device_discovery import Device, DeviceType, DeviceDiscovery
from palantir.network_mapper import NetworkGraph, NetworkMapper, RelationshipType
from palantir.shodan_client import ShodanResult


class TestDevice:
    """Tests for Device class."""
    
    def test_device_creation(self):
        """Test device creation."""
        device = Device(
            ip="192.168.1.1",
            port=80,
            service="http",
            banner="Apache/2.4.41",
            vulnerabilities=["CVE-2021-44228"],
            no_auth=True
        )
        
        assert device.ip == "192.168.1.1"
        assert device.port == 80
        assert device.service == "http"
        assert len(device.vulnerabilities) == 1
        assert device.no_auth is True
    
    def test_device_hashable(self):
        """Test that Device is hashable."""
        device1 = Device(ip="192.168.1.1", port=80, service="http")
        device2 = Device(ip="192.168.1.1", port=80, service="http")
        device3 = Device(ip="192.168.1.2", port=80, service="http")
        
        assert hash(device1) == hash(device2)
        assert hash(device1) != hash(device3)
        
        device_set = {device1, device2, device3}
        assert len(device_set) == 2
    
    def test_device_equality(self):
        """Test device equality."""
        device1 = Device(ip="192.168.1.1", port=80, service="http")
        device2 = Device(ip="192.168.1.1", port=80, service="http")
        device3 = Device(ip="192.168.1.1", port=443, service="https")
        
        assert device1 == device2
        assert device1 != device3


class TestDeviceDiscovery:
    """Tests for DeviceDiscovery class."""
    
    def test_classify_device_type(self):
        """Test device type classification."""
        discovery = DeviceDiscovery()
        
        # Test web server
        result = ShodanResult(
            ip="192.168.1.1",
            port=80,
            service="http",
            product="Apache"
        )
        device_type = discovery.classify_device_type(result)
        assert device_type == DeviceType.WEB_SERVER
        
        # Test SSH server
        result = ShodanResult(
            ip="192.168.1.1",
            port=22,
            service="ssh"
        )
        device_type = discovery.classify_device_type(result)
        assert device_type == DeviceType.SSH_SERVER
    
    def test_detect_no_auth(self):
        """Test no-auth detection."""
        discovery = DeviceDiscovery()
        
        # Test with no-auth keyword
        result = ShodanResult(
            ip="192.168.1.1",
            port=21,
            service="ftp",
            banner="220 Anonymous access granted"
        )
        assert discovery.detect_no_auth(result) is True
        
        # Test without no-auth keyword
        result = ShodanResult(
            ip="192.168.1.1",
            port=22,
            service="ssh",
            banner="SSH-2.0-OpenSSH"
        )
        assert discovery.detect_no_auth(result) is False
    
    def test_filter_devices(self):
        """Test device filtering."""
        discovery = DeviceDiscovery(prefer_no_auth=True)
        
        devices = [
            Device(ip="192.168.1.1", port=80, service="http", no_auth=True, vulnerabilities=["CVE-1"]),
            Device(ip="192.168.1.2", port=80, service="http", no_auth=False, vulnerabilities=[]),
            Device(ip="192.168.1.3", port=22, service="ssh", no_auth=True, vulnerabilities=["CVE-1", "CVE-2"]),
        ]
        
        # Filter by no_auth
        filtered = discovery.filter_devices(devices, {"no_auth": True})
        assert len(filtered) == 2
        assert all(d.no_auth for d in filtered)
        
        # Filter by min vulnerabilities
        filtered = discovery.filter_devices(devices, {"min_vulnerabilities": 1})
        assert len(filtered) == 2
        
        # Filter by service
        filtered = discovery.filter_devices(devices, {"service": "http"})
        assert len(filtered) == 2
        assert all("http" in d.service.lower() for d in filtered)


class TestNetworkMapper:
    """Tests for NetworkMapper class."""
    
    def test_same_subnet(self):
        """Test same-subnet detection."""
        mapper = NetworkMapper(subnet_threshold=24)
        
        device1 = Device(ip="192.168.1.10", port=80, service="http")
        device2 = Device(ip="192.168.1.20", port=80, service="http")
        device3 = Device(ip="192.168.2.10", port=80, service="http")
        
        assert mapper._same_subnet(device1, device2) is True
        assert mapper._same_subnet(device1, device3) is False
    
    def test_same_asn(self):
        """Test same-ASN detection."""
        mapper = NetworkMapper()
        
        device1 = Device(ip="192.168.1.1", port=80, service="http", asn="AS12345")
        device2 = Device(ip="192.168.1.2", port=80, service="http", asn="AS12345")
        device3 = Device(ip="192.168.1.3", port=80, service="http", asn="AS67890")
        
        assert mapper._same_asn(device1, device2) is True
        assert mapper._same_asn(device1, device3) is False
    
    def test_same_service(self):
        """Test same-service detection."""
        mapper = NetworkMapper()
        
        device1 = Device(ip="192.168.1.1", port=80, service="http")
        device2 = Device(ip="192.168.1.2", port=80, service="http")
        device3 = Device(ip="192.168.1.3", port=22, service="ssh")
        
        assert mapper._same_service(device1, device2) is True
        assert mapper._same_service(device1, device3) is False
    
    def test_map_network(self):
        """Test network mapping."""
        mapper = NetworkMapper()
        
        devices = [
            Device(ip="192.168.1.10", port=80, service="http", asn="AS12345"),
            Device(ip="192.168.1.20", port=80, service="http", asn="AS12345"),
            Device(ip="192.168.2.10", port=22, service="ssh", asn="AS67890"),
        ]
        
        network = mapper.map_network(devices)
        
        assert len(network.devices) == 3
        assert len(network.edges) > 0
        
        # Should have same-subnet and same-service relationships
        relationship_types = {rel for _, _, rel in network.edges}
        assert RelationshipType.SAME_SUBNET in relationship_types or \
               RelationshipType.SAME_SERVICE in relationship_types


class TestNetworkGraph:
    """Tests for NetworkGraph class."""
    
    def test_network_graph_creation(self):
        """Test network graph creation."""
        devices = [
            Device(ip="192.168.1.1", port=80, service="http"),
            Device(ip="192.168.1.2", port=80, service="http"),
        ]
        
        graph = NetworkGraph(devices=devices)
        
        assert len(graph.devices) == 2
        assert graph.get_device_by_ip("192.168.1.1") is not None
        assert graph.get_device_by_ip("192.168.1.99") is None
    
    def test_get_neighbors(self):
        """Test getting neighbors."""
        device1 = Device(ip="192.168.1.1", port=80, service="http")
        device2 = Device(ip="192.168.1.2", port=80, service="http")
        device3 = Device(ip="192.168.1.3", port=22, service="ssh")
        
        graph = NetworkGraph(
            devices=[device1, device2, device3],
            edges=[
                (device1, device2, RelationshipType.SAME_SERVICE),
                (device1, device3, RelationshipType.SAME_SUBNET),
            ]
        )
        
        neighbors = graph.get_neighbors(device1)
        assert len(neighbors) == 2
        assert device2 in neighbors
        assert device3 in neighbors


class TestCausalModeler:
    """Tests for CausalModeler class."""
    
    @patch('palantir.causal_modeler.CRCA_AVAILABLE', True)
    @patch('palantir.causal_modeler.CRCAAgent')
    def test_causal_modeler_initialization(self, mock_crca):
        """Test causal modeler initialization."""
        from palantir.causal_modeler import CausalModeler
        
        mock_agent = MagicMock()
        mock_crca.return_value = mock_agent
        
        modeler = CausalModeler()
        
        assert modeler.crca_agent is not None
        # Should have added causal relationships
        assert mock_agent.add_causal_relationship.called
    
    @patch('palantir.causal_modeler.CRCA_AVAILABLE', True)
    @patch('palantir.causal_modeler.CRCAAgent')
    def test_calculate_vulnerability_score(self, mock_crca):
        """Test vulnerability score calculation."""
        from palantir.causal_modeler import CausalModeler
        
        mock_agent = MagicMock()
        mock_crca.return_value = mock_agent
        
        modeler = CausalModeler()
        
        # Device with vulnerabilities
        device = Device(
            ip="192.168.1.1",
            port=80,
            service="http",
            vulnerabilities=["CVE-1", "CVE-2"],
            no_auth=True
        )
        
        score = modeler.calculate_device_vulnerability_score(device)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high with 2 vulns and no auth


class TestShodanClient:
    """Tests for ShodanClient class."""
    
    @patch('palantir.shodan_client.SHODAN_AVAILABLE', True)
    @patch('palantir.shodan_client.shodan')
    def test_shodan_client_initialization(self, mock_shodan_module):
        """Test Shodan client initialization."""
        from palantir.shodan_client import ShodanClient
        
        mock_api = MagicMock()
        mock_shodan_module.Shodan.return_value = mock_api
        
        client = ShodanClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.api is not None
    
    @patch('palantir.shodan_client.SHODAN_AVAILABLE', True)
    @patch('palantir.shodan_client.shodan')
    def test_rate_limiting(self, mock_shodan_module):
        """Test rate limiting."""
        from palantir.shodan_client import ShodanClient
        import time
        
        mock_api = MagicMock()
        mock_shodan_module.Shodan.return_value = mock_api
        
        client = ShodanClient(api_key="test-key", rate_limit=10.0)
        
        start_time = time.time()
        client._rate_limit_wait()
        client._rate_limit_wait()
        elapsed = time.time() - start_time
        
        # Should have waited at least 0.1 seconds (1/10)
        assert elapsed >= 0.05  # Allow some tolerance


def test_integration_mock():
    """Integration test with mocked Shodan."""
    from palantir.palantir_agent import PalantirAgent
    
    # Mock Shodan client
    with patch('palantir.shodan_client.SHODAN_AVAILABLE', True), \
         patch('palantir.shodan_client.shodan') as mock_shodan:
        
        mock_api = MagicMock()
        mock_shodan.Shodan.return_value = mock_api
        
        # Mock search results
        mock_api.search.return_value = {
            "matches": [
                {
                    "ip_str": "192.168.1.1",
                    "port": 80,
                    "product": "Apache",
                    "data": "HTTP/1.1 200 OK",
                    "hostnames": [],
                    "location": {},
                    "org": "Test Org",
                    "asn": "AS12345",
                    "os": "Linux",
                    "vulns": {"CVE-2021-44228": {}}
                }
            ]
        }
        
        # Create agent
        agent = PalantirAgent(shodan_api_key="test-key")
        
        # Test discovery
        devices = agent.discover_devices(
            query="product:Apache",
            filters={"max_results": 1}
        )
        
        assert len(devices) > 0
        assert devices[0].ip == "192.168.1.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

