"""
Palantir Usage Examples

Demonstrates how to use Palantir for security intelligence gathering.
"""

import os
from pathlib import Path
from loguru import logger

# Set up logging
logger.add("palantir_example.log", rotation="10 MB")

try:
    from palantir import PalantirAgent
    from palantir.device_discovery import DeviceType
except ImportError:
    logger.error("Palantir module not found. Make sure it's installed.")
    exit(1)


def example_basic_discovery():
    """Example: Basic device discovery."""
    print("\n=== Example 1: Basic Device Discovery ===")
    
    # Initialize Palantir
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key, prefer_no_auth=True)
    
    # Discover Apache web servers with no authentication
    print("Discovering Apache web servers...")
    devices = palantir.discover_devices(
        query="product:Apache httpd",
        filters={
            "no_auth": True,
            "max_results": 20
        }
    )
    
    print(f"Discovered {len(devices)} devices")
    for device in devices[:5]:
        print(f"  - {device.ip}:{device.port} ({device.service}) - "
              f"{len(device.vulnerabilities)} vulnerabilities, "
              f"no_auth={device.no_auth}")


def example_network_mapping():
    """Example: Network topology mapping."""
    print("\n=== Example 2: Network Topology Mapping ===")
    
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key)
    
    # Discover devices
    print("Discovering devices...")
    devices = palantir.discover_devices(
        query="port:22",
        filters={"max_results": 30}
    )
    
    if not devices:
        print("No devices found")
        return
    
    # Map network
    print("Mapping network topology...")
    network = palantir.map_network(devices)
    
    print(f"Network mapped:")
    print(f"  - Devices: {len(network.devices)}")
    print(f"  - Relationships: {len(network.edges)}")
    print(f"  - Communities: {len(network.communities)}")
    
    # Show some relationships
    print("\nSample relationships:")
    for source, target, rel_type in network.edges[:5]:
        print(f"  - {source.ip} <-> {target.ip} ({rel_type.value})")


def example_attack_path_analysis():
    """Example: Attack path analysis."""
    print("\n=== Example 3: Attack Path Analysis ===")
    
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key)
    
    # Discover and map network
    print("Discovering devices and mapping network...")
    devices = palantir.discover_devices(
        query="product:Apache httpd",
        filters={"max_results": 20}
    )
    
    if len(devices) < 2:
        print("Need at least 2 devices for path analysis")
        return
    
    network = palantir.map_network(devices)
    
    # Analyze attack paths
    print("Analyzing attack paths...")
    attack_paths = palantir.analyze_attack_paths(network_graph=network)
    
    print(f"Found {len(attack_paths)} attack paths")
    
    # Show top paths
    print("\nTop 3 attack paths:")
    for i, path in enumerate(attack_paths[:3], 1):
        path_ips = [d.ip for d in path.path]
        print(f"  {i}. {' -> '.join(path_ips)}")
        print(f"     Risk: {path.risk_score:.2f}, Probability: {path.probability:.2f}")
        print(f"     {path.reasoning}")


def example_threat_scenarios():
    """Example: Threat scenario generation."""
    print("\n=== Example 4: Threat Scenario Generation ===")
    
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key)
    
    # Discover and map network
    print("Discovering devices and mapping network...")
    devices = palantir.discover_devices(
        query="port:22",
        filters={"max_results": 15}
    )
    
    if not devices:
        print("No devices found")
        return
    
    network = palantir.map_network(devices)
    
    # Generate threat scenarios
    print("Generating threat scenarios...")
    scenarios = palantir.generate_threat_scenarios(
        network_graph=network,
        n_scenarios=5
    )
    
    print(f"Generated {len(scenarios)} threat scenarios")
    
    # Show scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario.name}")
        print(f"  Entry: {scenario.entry_point.ip}")
        print(f"  Target: {scenario.target_device.ip}")
        print(f"  Probability: {scenario.probability:.2f}")
        print(f"  Path length: {len(scenario.attack_path)} devices")
        print(f"  Reasoning: {scenario.reasoning[:100]}...")


def example_visualization():
    """Example: Network visualization."""
    print("\n=== Example 5: Network Visualization ===")
    
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key)
    
    # Discover and map network
    print("Discovering devices and mapping network...")
    devices = palantir.discover_devices(
        query="product:Apache httpd",
        filters={"max_results": 15}
    )
    
    if not devices:
        print("No devices found")
        return
    
    network = palantir.map_network(devices)
    
    # Analyze attack paths
    attack_paths = palantir.analyze_attack_paths(network_graph=network)
    
    # Create output directory
    output_dir = Path("palantir_output")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize
    print("Generating visualizations...")
    try:
        palantir.visualize_network(
            network_graph=network,
            attack_paths=attack_paths[:5],
            output_file=str(output_dir / "network_topology.png")
        )
        print(f"  - Network topology saved to {output_dir / 'network_topology.png'}")
        
        if attack_paths:
            palantir.visualize_attack_paths(
                attack_paths=attack_paths[:10],
                output_file=str(output_dir / "attack_paths.png")
            )
            print(f"  - Attack paths saved to {output_dir / 'attack_paths.png'}")
        
        palantir.visualize_vulnerability_heatmap(
            network_graph=network,
            output_file=str(output_dir / "vulnerability_heatmap.png")
        )
        print(f"  - Vulnerability heatmap saved to {output_dir / 'vulnerability_heatmap.png'}")
        
    except Exception as e:
        print(f"Visualization error: {e}")


def example_full_analysis():
    """Example: Full analysis pipeline."""
    print("\n=== Example 6: Full Analysis Pipeline ===")
    
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        print("ERROR: SHODAN_API_KEY environment variable not set")
        return
    
    palantir = PalantirAgent(shodan_api_key=api_key)
    
    # Run full analysis
    print("Running full analysis pipeline...")
    results = palantir.full_analysis(
        query="product:Apache httpd",
        filters={"no_auth": True, "max_results": 20},
        max_results=20,
        n_scenarios=5,
        output_dir="palantir_output"
    )
    
    print("\nAnalysis Results:")
    print(f"  - Devices discovered: {results.get('devices_discovered', 0)}")
    print(f"  - Network size: {results.get('network_size', 0)}")
    print(f"  - Relationships: {results.get('relationships', 0)}")
    print(f"  - Communities: {results.get('communities', 0)}")
    print(f"  - Attack paths: {results.get('attack_paths', 0)}")
    print(f"  - Threat scenarios: {results.get('threat_scenarios', 0)}")
    
    print("\nTop Attack Paths:")
    for i, path in enumerate(results.get('top_attack_paths', [])[:3], 1):
        print(f"  {i}. {' -> '.join(path['path'])}")
        print(f"     Risk: {path['risk_score']:.2f}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Palantir Intelligence System - Usage Examples")
    print("=" * 60)
    print("\nWARNING: This tool is for DEFENSIVE SECURITY RESEARCH ONLY.")
    print("Use only on systems you own or have explicit authorization to test.\n")
    
    # Check for API key
    if not os.getenv("SHODAN_API_KEY"):
        print("\nERROR: SHODAN_API_KEY environment variable not set")
        print("Please set it before running examples:")
        print("  export SHODAN_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run examples
        example_basic_discovery()
        example_network_mapping()
        example_attack_path_analysis()
        example_threat_scenarios()
        example_visualization()
        example_full_analysis()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        
    except Exception as e:
        logger.exception("Error running examples")
        print(f"\nError: {e}")
        print("Check palantir_example.log for details")


if __name__ == "__main__":
    main()

