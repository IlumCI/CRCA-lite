"""
Graph management module for specialized agents.

Provides graph operations including node/edge management, topological sorting,
path finding, and DAG validation. Supports both dict-based and rustworkx-based graphs.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Manages graph operations for specialized agents.
    
    Supports both dict-based graphs (always available) and rustworkx-based
    graphs (when rustworkx is installed). Falls back gracefully to dict-only
    operations when rustworkx is unavailable.
    """

    def __init__(self, graph_type: str = "causal"):
        """
        Initialize the graph manager.
        
        Args:
            graph_type: Type of graph (causal, knowledge, dependency, etc.)
        """
        self.graph_type = graph_type
        self.graph: Dict[str, Dict[str, Any]] = {}
        self.graph_reverse: Dict[str, List[str]] = {}  # For fast parent lookup
        
        # Rustworkx graph (optional)
        self._rustworkx_graph = None
        self._node_to_index: Dict[str, int] = {}
        self._index_to_node: Dict[int, str] = {}
        
        if RUSTWORKX_AVAILABLE:
            try:
                self._rustworkx_graph = rx.PyDiGraph()
            except Exception:
                logger.warning("Failed to initialize rustworkx graph, using dict-only mode")
                self._rustworkx_graph = None

    def ensure_node_exists(self, node: str) -> None:
        """
        Ensure a node exists in the graph.
        
        Args:
            node: Node identifier
        """
        if node not in self.graph:
            self.graph[node] = {}
        if node not in self.graph_reverse:
            self.graph_reverse[node] = []
        try:
            self._ensure_node_index(node)
        except Exception:
            pass

    def add_relationship(
        self,
        source: str,
        target: str,
        strength: float = 1.0,
        relation_type: Optional[Any] = None,
        confidence: float = 1.0,
        **metadata
    ) -> None:
        """
        Add a relationship between two nodes.
        
        Args:
            source: Source node
            target: Target node
            strength: Relationship strength
            relation_type: Type of relationship (can be Enum or string)
            confidence: Confidence in the relationship
            **metadata: Additional metadata to store
        """
        self.ensure_node_exists(source)
        self.ensure_node_exists(target)

        meta = {
            "strength": float(strength),
            "confidence": float(confidence),
            **metadata
        }
        
        if relation_type is not None:
            if isinstance(relation_type, Enum):
                meta["relation_type"] = relation_type.value
            else:
                meta["relation_type"] = str(relation_type)

        self.graph.setdefault(source, {})[target] = meta

        if source not in self.graph_reverse.get(target, []):
            self.graph_reverse.setdefault(target, []).append(source)

        # Update rustworkx graph if available
        if self._rustworkx_graph is not None:
            try:
                u_idx = self._ensure_node_index(source)
                v_idx = self._ensure_node_index(target)
                try:
                    existing = self._rustworkx_graph.get_edge_data(u_idx, v_idx)
                except Exception:
                    existing = None

                if existing is None:
                    try:
                        self._rustworkx_graph.add_edge(u_idx, v_idx, meta)
                    except Exception:
                        logger.debug(
                            f"rustworkx.add_edge failed for {source}->{target}; continuing with dict-only graph."
                        )
                else:
                    try:
                        if isinstance(existing, dict):
                            existing.update(meta)
                        else:
                            try:
                                edge_idx = self._rustworkx_graph.get_edge_index(u_idx, v_idx)
                            except Exception:
                                edge_idx = None
                            if edge_idx is not None and edge_idx >= 0:
                                try:
                                    self._rustworkx_graph.remove_edge(edge_idx)
                                    self._rustworkx_graph.add_edge(u_idx, v_idx, meta)
                                except Exception:
                                    logger.debug(
                                        f"Could not replace rustworkx edge for {source}->{target}; keeping dict-only metadata."
                                    )
                    except Exception:
                        logger.debug(
                            f"Failed updating rustworkx edge for {source}->{target}; continuing with dict-only graph."
                        )
            except Exception:
                logger.debug(
                    "rustworkx operation failed during add_relationship; continuing with dict-only graph."
                )

    def get_parents(self, node: str) -> List[str]:
        """
        Get parent nodes of a given node.
        
        Args:
            node: Node identifier
            
        Returns:
            List of parent node identifiers
        """
        return self.graph_reverse.get(node, [])

    def get_children(self, node: str) -> List[str]:
        """
        Get child nodes of a given node.
        
        Args:
            node: Node identifier
            
        Returns:
            List of child node identifiers
        """
        return list(self.graph.get(node, {}).keys())

    def _ensure_node_index(self, name: str) -> int:
        """
        Ensure a node has an index in the rustworkx graph.
        
        Args:
            name: Node identifier
            
        Returns:
            Node index
        """
        if name in self._node_to_index:
            return self._node_to_index[name]
        if self._rustworkx_graph is None:
            raise RuntimeError("rustworkx graph not available")
        idx = self._rustworkx_graph.add_node(name)
        self._node_to_index[name] = idx
        self._index_to_node[idx] = name
        return idx

    def _node_index(self, name: str) -> Optional[int]:
        """
        Get the rustworkx index for a node.
        
        Args:
            name: Node identifier
            
        Returns:
            Node index or None if not found
        """
        return self._node_to_index.get(name)

    def _node_name(self, idx: int) -> Optional[str]:
        """
        Get the node name for a rustworkx index.
        
        Args:
            idx: Node index
            
        Returns:
            Node identifier or None if not found
        """
        return self._index_to_node.get(idx)

    def edge_strength(self, source: str, target: str) -> float:
        """
        Get the strength of an edge.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Edge strength (0.0 if edge doesn't exist)
        """
        edge = self.graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0

    def topological_sort(self) -> List[str]:
        """
        Perform topological sort of the graph.
        
        Returns:
            List of nodes in topological order
        """
        if self._rustworkx_graph is not None:
            try:
                order_idx = rx.topological_sort(self._rustworkx_graph)
                result = [self._node_name(i) for i in order_idx if self._node_name(i) is not None]
                for n in list(self.graph.keys()):
                    if n not in result:
                        result.append(n)
                return result
            except Exception:
                pass
        
        # Fallback to dict-based topological sort
        in_degree: Dict[str, int] = {node: 0 for node in self.graph.keys()}
        for node in self.graph:
            for child in self.get_children(node):
                in_degree[child] = in_degree.get(child, 0) + 1

        queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
        result: List[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def identify_path(self, start: str, end: str) -> List[str]:
        """
        Find a path from start to end node.
        
        Args:
            start: Starting node
            end: Ending node
            
        Returns:
            List of nodes forming the path, or empty list if no path exists
        """
        if start not in self.graph or end not in self.graph:
            return []

        if start == end:
            return [start]

        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: set = {start}

        while queue:
            current, path = queue.pop(0)

            for child in self.get_children(current):
                if child == end:
                    return path + [child]

                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        return []  # No path found

    def has_path(self, start: str, end: str) -> bool:
        """
        Check if a path exists from start to end.
        
        Args:
            start: Starting node
            end: Ending node
            
        Returns:
            True if path exists, False otherwise
        """
        if start == end:
            return True

        stack = [start]
        visited = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            for child in self.get_children(current):
                if child == end:
                    return True
                if child not in visited:
                    stack.append(child)

        return False

    def get_descendants(self, node: str) -> List[str]:
        """
        Get all descendant nodes of a given node.
        
        Args:
            node: Node identifier
            
        Returns:
            List of descendant node identifiers
        """
        if node not in self.graph:
            return []
        stack = [node]
        visited = set()
        descendants: List[str] = []
        while stack:
            cur = stack.pop()
            for child in self.get_children(cur):
                if child in visited:
                    continue
                visited.add(child)
                descendants.append(child)
                stack.append(child)
        return descendants

    def get_nodes(self) -> List[str]:
        """
        Get all nodes in the graph.
        
        Returns:
            List of node identifiers
        """
        return list(self.graph.keys())

    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Get all edges in the graph.
        
        Returns:
            List of (source, target) tuples
        """
        edges = []
        for source, targets in self.graph.items():
            for target in targets.keys():
                edges.append((source, target))
        return edges

    def is_dag(self) -> bool:
        """
        Check if the graph is a directed acyclic graph (DAG).
        
        Returns:
            True if DAG, False otherwise
        """
        if self._rustworkx_graph is not None:
            try:
                return rx.is_directed_acyclic_graph(self._rustworkx_graph)
            except Exception:
                pass

        # Fallback to dict-based cycle detection
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for child in self.get_children(node):
                if child not in visited:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        rec_stack = set()

        for node in self.graph:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    return False

        return True

    def add_nodes_from(self, nodes: List[str]) -> None:
        """
        Add multiple nodes to the graph.
        
        Args:
            nodes: List of node identifiers
        """
        for n in nodes:
            self.ensure_node_exists(n)

    def add_edges_from(self, edges: List[Tuple[str, str]], **default_metadata) -> None:
        """
        Add multiple edges to the graph.
        
        Args:
            edges: List of (source, target) tuples
            **default_metadata: Default metadata for edges
        """
        for u, v in edges:
            self.add_relationship(u, v, **default_metadata)

