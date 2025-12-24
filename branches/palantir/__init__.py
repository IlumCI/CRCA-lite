"""
Palantir: Intelligent Security Intelligence System

A defensive security research tool that integrates Shodan.io device discovery
with CRCA's causal reasoning engine to map vulnerable device networks,
model attack paths, and visualize threat surfaces.

WARNING: This tool is for DEFENSIVE SECURITY RESEARCH ONLY.
Use only on systems you own or have explicit written authorization to test.
"""

from palantir.palantir_agent import PalantirAgent
from palantir.device_discovery import Device, DeviceType
from palantir.network_mapper import NetworkGraph, RelationshipType

__all__ = [
    "PalantirAgent",
    "Device",
    "DeviceType",
    "NetworkGraph",
    "RelationshipType",
]

__version__ = "1.0.0"

