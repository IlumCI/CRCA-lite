"""
Device Discovery and Classification

Filters and classifies devices discovered via Shodan.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from loguru import logger

from palantir.shodan_client import ShodanResult


class DeviceType(Enum):
    """Device type classification."""
    
    WEB_SERVER = "web_server"
    DATABASE = "database"
    SSH_SERVER = "ssh_server"
    FTP_SERVER = "ftp_server"
    TELNET_SERVER = "telnet_server"
    IOT_DEVICE = "iot_device"
    ROUTER = "router"
    CAMERA = "camera"
    PRINTER = "printer"
    UNKNOWN = "unknown"


@dataclass
class Device:
    """
    Represents a discovered device with metadata.
    
    Args:
        ip: IP address
        port: Port number
        service: Service name
        banner: Service banner/response
        vulnerabilities: List of CVE identifiers
        no_auth: Whether device requires no authentication
        location: Geographic location data
        device_type: Classified device type
        metadata: Additional metadata
    """
    
    ip: str
    port: int
    service: str
    banner: Optional[str] = None
    vulnerabilities: List[str] = field(default_factory=list)
    no_auth: bool = False
    location: Optional[Dict] = None
    device_type: DeviceType = DeviceType.UNKNOWN
    hostnames: List[str] = field(default_factory=list)
    org: Optional[str] = None
    asn: Optional[str] = None
    os: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Make Device hashable for use in sets."""
        return hash((self.ip, self.port))
    
    def __eq__(self, other: object) -> bool:
        """Compare devices by IP and port."""
        if not isinstance(other, Device):
            return False
        return self.ip == other.ip and self.port == other.port


class DeviceDiscovery:
    """
    Discovers and classifies devices from Shodan results.
    
    Args:
        prefer_no_auth: Prefer devices with no authentication (default: True)
        min_vulnerabilities: Minimum number of vulnerabilities to include (default: 0)
    """
    
    # Service to device type mapping
    SERVICE_TYPE_MAP: Dict[str, DeviceType] = {
        "http": DeviceType.WEB_SERVER,
        "https": DeviceType.WEB_SERVER,
        "apache": DeviceType.WEB_SERVER,
        "nginx": DeviceType.WEB_SERVER,
        "iis": DeviceType.WEB_SERVER,
        "ssh": DeviceType.SSH_SERVER,
        "mysql": DeviceType.DATABASE,
        "postgresql": DeviceType.DATABASE,
        "mongodb": DeviceType.DATABASE,
        "redis": DeviceType.DATABASE,
        "ftp": DeviceType.FTP_SERVER,
        "telnet": DeviceType.TELNET_SERVER,
        "rtsp": DeviceType.CAMERA,
        "onvif": DeviceType.CAMERA,
        "printer": DeviceType.PRINTER,
        "router": DeviceType.ROUTER,
    }
    
    # Keywords indicating no authentication
    NO_AUTH_KEYWORDS: Set[str] = {
        "authentication disabled",
        "no authentication",
        "anonymous",
        "public",
        "guest",
        "default password",
        "weak password",
    }
    
    def __init__(
        self,
        prefer_no_auth: bool = True,
        min_vulnerabilities: int = 0
    ) -> None:
        """
        Initialize device discovery.
        
        Args:
            prefer_no_auth: Prefer devices with no authentication.
            min_vulnerabilities: Minimum number of vulnerabilities to include.
        """
        self.prefer_no_auth = prefer_no_auth
        self.min_vulnerabilities = min_vulnerabilities
        logger.info(
            f"Device discovery initialized (prefer_no_auth={prefer_no_auth}, "
            f"min_vulnerabilities={min_vulnerabilities})"
        )
    
    def classify_device_type(self, result: ShodanResult) -> DeviceType:
        """
        Classify device type from Shodan result.
        
        Args:
            result: Shodan result to classify
            
        Returns:
            DeviceType classification
        """
        service_lower = result.service.lower()
        product_lower = (result.product or "").lower()
        banner_lower = (result.banner or "").lower()
        
        # Check service name
        for service_key, device_type in self.SERVICE_TYPE_MAP.items():
            if service_key in service_lower:
                return device_type
        
        # Check product name
        for service_key, device_type in self.SERVICE_TYPE_MAP.items():
            if service_key in product_lower:
                return device_type
        
        # Check banner for specific patterns
        if any(keyword in banner_lower for keyword in ["camera", "ip camera", "webcam"]):
            return DeviceType.CAMERA
        
        if any(keyword in banner_lower for keyword in ["router", "gateway", "access point"]):
            return DeviceType.ROUTER
        
        if any(keyword in banner_lower for keyword in ["printer", "hp", "canon", "epson"]):
            return DeviceType.PRINTER
        
        # IoT device heuristics
        if any(keyword in banner_lower for keyword in ["iot", "smart", "embedded"]):
            return DeviceType.IOT_DEVICE
        
        return DeviceType.UNKNOWN
    
    def detect_no_auth(self, result: ShodanResult) -> bool:
        """
        Detect if device requires no authentication.
        
        Args:
            result: Shodan result to check
            
        Returns:
            True if device appears to require no authentication
        """
        banner_lower = (result.banner or "").lower()
        
        # Check for no-auth keywords
        if any(keyword in banner_lower for keyword in self.NO_AUTH_KEYWORDS):
            return True
        
        # Check for anonymous access patterns
        if "anonymous" in banner_lower and "login" in banner_lower:
            return True
        
        # Check metadata for authentication fields
        metadata = result.metadata
        if isinstance(metadata, dict):
            # Some services expose auth status
            if metadata.get("authentication") == False:
                return True
        
        return False
    
    def convert_shodan_result(self, result: ShodanResult) -> Device:
        """
        Convert ShodanResult to Device.
        
        Args:
            result: Shodan result to convert
            
        Returns:
            Device object
        """
        device = Device(
            ip=result.ip,
            port=result.port,
            service=result.service,
            banner=result.banner,
            vulnerabilities=result.vulns.copy(),
            no_auth=self.detect_no_auth(result),
            location=result.location,
            device_type=self.classify_device_type(result),
            hostnames=result.hostnames.copy(),
            org=result.org,
            asn=result.asn,
            os=result.os,
            product=result.product,
            version=result.version,
            metadata=result.metadata.copy() if isinstance(result.metadata, dict) else {}
        )
        
        return device
    
    def filter_devices(
        self,
        devices: List[Device],
        filters: Optional[Dict] = None
    ) -> List[Device]:
        """
        Filter devices based on criteria.
        
        Args:
            devices: List of devices to filter
            filters: Filter criteria dictionary
            
        Returns:
            Filtered list of devices
        """
        if not filters:
            filters = {}
        
        filtered = devices.copy()
        
        # Filter by no-auth preference
        if filters.get("no_auth", self.prefer_no_auth):
            filtered = [d for d in filtered if d.no_auth]
        
        # Filter by minimum vulnerabilities
        min_vulns = filters.get("min_vulnerabilities", self.min_vulnerabilities)
        if min_vulns > 0:
            filtered = [d for d in filtered if len(d.vulnerabilities) >= min_vulns]
        
        # Filter by device type
        if "device_type" in filters:
            device_type = filters["device_type"]
            if isinstance(device_type, DeviceType):
                filtered = [d for d in filtered if d.device_type == device_type]
            elif isinstance(device_type, list):
                filtered = [d for d in filtered if d.device_type in device_type]
        
        # Filter by service
        if "service" in filters:
            service = filters["service"].lower()
            filtered = [d for d in filtered if service in d.service.lower()]
        
        # Filter by vulnerability CVE
        if "cve" in filters:
            cve = filters["cve"].upper()
            filtered = [
                d for d in filtered
                if any(cve in vuln.upper() for vuln in d.vulnerabilities)
            ]
        
        # Filter by IP range
        if "ip_range" in filters:
            ip_range = filters["ip_range"]
            try:
                network = ipaddress.ip_network(ip_range, strict=False)
                filtered = [
                    d for d in filtered
                    if ipaddress.ip_address(d.ip) in network
                ]
            except ValueError:
                logger.warning(f"Invalid IP range: {ip_range}")
        
        # Limit results
        max_results = filters.get("max_results")
        if max_results:
            filtered = filtered[:max_results]
        
        logger.info(f"Filtered {len(devices)} devices to {len(filtered)}")
        return filtered
    
    def discover_devices(
        self,
        shodan_results: List[ShodanResult],
        filters: Optional[Dict] = None
    ) -> List[Device]:
        """
        Discover and classify devices from Shodan results.
        
        Args:
            shodan_results: List of Shodan results
            filters: Optional filter criteria
            
        Returns:
            List of classified and filtered devices
        """
        # Convert Shodan results to Devices
        devices = [self.convert_shodan_result(result) for result in shodan_results]
        
        # Apply filters
        filtered = self.filter_devices(devices, filters)
        
        logger.info(
            f"Discovered {len(filtered)} devices "
            f"(from {len(shodan_results)} Shodan results)"
        )
        
        return filtered

