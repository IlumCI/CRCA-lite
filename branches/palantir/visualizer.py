"""
Network and Attack Visualization

Visualizes network topology, attack paths, and vulnerability heatmaps.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, visualization disabled")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from palantir.device_discovery import Device, DeviceType
from palantir.network_mapper import NetworkGraph
from palantir.attack_path_analyzer import AttackPath, ThreatScenario


class Visualizer:
    """
    Visualizes network topology and attack paths.
    
    Args:
        figure_size: Figure size tuple (width, height) in inches
        dpi: Dots per inch for output
    """
    
    def __init__(
        self,
        figure_size: tuple = (16, 12),
        dpi: int = 100
    ) -> None:
        """
        Initialize visualizer.
        
        Args:
            figure_size: Figure size tuple (width, height).
            dpi: Dots per inch for output.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")
        
        self.figure_size = figure_size
        self.dpi = dpi
        logger.info("Visualizer initialized")
    
    def visualize_network(
        self,
        network: NetworkGraph,
        attack_paths: Optional[List[AttackPath]] = None,
        output_file: Optional[str] = None,
        show_labels: bool = True,
        layout: str = "spring"
    ) -> None:
        """
        Visualize network topology with optional attack paths.
        
        Args:
            network: Network graph to visualize
            attack_paths: Optional attack paths to highlight
            output_file: Optional output file path
            show_labels: Whether to show device labels
            layout: Layout algorithm ("spring", "circular", "kamada_kawai")
        """
        if not NETWORKX_AVAILABLE or not network.graph:
            logger.warning("NetworkX graph not available, cannot visualize")
            return
        
        logger.info(f"Visualizing network with {len(network.devices)} devices")
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(network.graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(network.graph)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(network.graph)
            except Exception:
                pos = nx.spring_layout(network.graph)
        else:
            pos = nx.spring_layout(network.graph)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        
        for edge in network.graph.edges(data=True):
            rel_type = edge[2].get("relationship", "unknown")
            edge_colors.append(self._get_edge_color(rel_type))
            edge_widths.append(0.5)
        
        nx.draw_networkx_edges(
            network.graph,
            pos,
            ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6
        )
        
        # Highlight attack paths
        if attack_paths:
            for path_obj in attack_paths[:5]:  # Top 5 paths
                path_edges = [
                    (path_obj.path[i], path_obj.path[i + 1])
                    for i in range(len(path_obj.path) - 1)
                ]
                nx.draw_networkx_edges(
                    network.graph,
                    pos,
                    edgelist=path_edges,
                    ax=ax,
                    edge_color="red",
                    width=3.0,
                    alpha=0.8,
                    style="dashed"
                )
        
        # Draw nodes with colors based on vulnerability
        node_colors = []
        node_sizes = []
        
        for device in network.devices:
            vuln_count = len(device.vulnerabilities)
            risk = min(vuln_count * 0.2 + (0.3 if device.no_auth else 0.0), 1.0)
            node_colors.append(risk)
            node_sizes.append(300 + vuln_count * 50)
        
        nodes = nx.draw_networkx_nodes(
            network.graph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.Reds,
            alpha=0.8
        )
        
        # Draw labels
        if show_labels:
            labels = {device: f"{device.ip}\n{device.port}" for device in network.devices}
            nx.draw_networkx_labels(
                network.graph,
                pos,
                labels,
                ax=ax,
                font_size=8
            )
        
        # Add legend
        self._add_legend(ax, network, attack_paths is not None)
        
        ax.set_title("Palantir Network Topology", fontsize=16, fontweight="bold")
        ax.axis("off")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Network visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_attack_paths(
        self,
        attack_paths: List[AttackPath],
        output_file: Optional[str] = None
    ) -> None:
        """
        Visualize attack paths as a directed graph.
        
        Args:
            attack_paths: List of attack paths
            output_file: Optional output file path
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, cannot visualize paths")
            return
        
        logger.info(f"Visualizing {len(attack_paths)} attack paths")
        
        # Create directed graph from paths
        G = nx.DiGraph()
        
        for path_obj in attack_paths:
            for i in range(len(path_obj.path) - 1):
                source = path_obj.path[i]
                target = path_obj.path[i + 1]
                
                if not G.has_edge(source, target):
                    G.add_edge(source, target, risk=path_obj.risk_score)
                else:
                    # Update risk if higher
                    current_risk = G[source][target].get("risk", 0.0)
                    if path_obj.risk_score > current_risk:
                        G[source][target]["risk"] = path_obj.risk_score
        
        if not G.nodes():
            logger.warning("No nodes in attack path graph")
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges with colors based on risk
        edge_colors = [G[u][v].get("risk", 0.0) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Reds,
            width=2.0,
            alpha=0.7,
            arrows=True,
            arrowsize=20
        )
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            vuln_count = len(node.vulnerabilities)
            risk = min(vuln_count * 0.2 + (0.3 if node.no_auth else 0.0), 1.0)
            node_colors.append(risk)
        
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=500,
            cmap=plt.cm.Reds,
            alpha=0.8
        )
        
        # Draw labels
        labels = {node: f"{node.ip}:{node.port}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title("Attack Paths", fontsize=16, fontweight="bold")
        ax.axis("off")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Attack paths visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_vulnerability_heatmap(
        self,
        network: NetworkGraph,
        output_file: Optional[str] = None
    ) -> None:
        """
        Create vulnerability heatmap of the network.
        
        Args:
            network: Network graph
            output_file: Optional output file path
        """
        logger.info("Creating vulnerability heatmap")
        
        # Calculate vulnerability scores
        device_scores: List[float] = []
        device_labels: List[str] = []
        
        for device in network.devices:
            score = len(device.vulnerabilities) + (1.0 if device.no_auth else 0.0)
            device_scores.append(score)
            device_labels.append(f"{device.ip}:{device.port}")
        
        if not device_scores:
            logger.warning("No devices to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(max(12, len(device_scores) * 0.5), 8), dpi=self.dpi)
        
        # Create heatmap
        y_pos = range(len(device_scores))
        colors = plt.cm.Reds([s / max(device_scores) if device_scores else 0.0 for s in device_scores])
        
        bars = ax.barh(y_pos, device_scores, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(device_labels, fontsize=8)
        ax.set_xlabel("Vulnerability Score", fontsize=12)
        ax.set_title("Device Vulnerability Heatmap", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Vulnerability heatmap saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_edge_color(self, relationship_type: str) -> str:
        """Get color for relationship type."""
        color_map = {
            "same_subnet": "blue",
            "same_asn": "green",
            "same_org": "purple",
            "same_service": "orange",
            "same_vulnerability": "red",
            "geographic_proximity": "cyan",
            "service_dependency": "magenta"
        }
        return color_map.get(relationship_type, "gray")
    
    def _add_legend(
        self,
        ax: Any,
        network: NetworkGraph,
        has_attack_paths: bool
    ) -> None:
        """Add legend to network visualization."""
        legend_elements = [
            mpatches.Patch(color="red", alpha=0.8, label="High Risk Device"),
            mpatches.Patch(color="orange", alpha=0.8, label="Medium Risk Device"),
            mpatches.Patch(color="yellow", alpha=0.8, label="Low Risk Device"),
        ]
        
        if has_attack_paths:
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color="red", linewidth=3, linestyle="--", label="Attack Path")
            )
        
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

