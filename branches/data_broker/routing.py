"""
Data Routing Engine

Uses causal graphs to intelligently route data between producers and consumers.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class RouteMatchStrategy(Enum):
    """Strategies for matching data routes."""
    CAUSAL_ONLY = "causal_only"
    SCHEMA_ONLY = "schema_only"
    QUALITY_ONLY = "quality_only"
    COMPOSITE = "composite"


@dataclass
class RouteMatch:
    """Represents a match between a data producer and consumer."""
    
    producer: str
    consumer: str
    confidence: float
    causal_strength: float = 0.0
    schema_compatibility: float = 0.0
    quality_score: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsumerRequirement:
    """Requirements for a data consumer."""
    
    name: str
    required_fields: List[str] = field(default_factory=list)
    min_quality_score: float = 0.0
    causal_dependencies: List[str] = field(default_factory=list)
    schema_preferences: Dict[str, str] = field(default_factory=dict)
    description: str = ""


class RoutingEngine:
    """Engine for routing data using causal graphs and schema matching."""
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, Dict[str, float]]] = None,
        strategy: RouteMatchStrategy = RouteMatchStrategy.COMPOSITE
    ) -> None:
        """
        Initialize the routing engine.
        
        Args:
            causal_graph: Causal graph dictionary mapping sources to targets with strengths
            strategy: Strategy for matching routes
        """
        self.causal_graph = causal_graph or {}
        self.strategy = strategy
        self.consumers: Dict[str, ConsumerRequirement] = {}
        self.producers: Dict[str, Dict[str, Any]] = {}
        
    def register_consumer(self, requirement: ConsumerRequirement) -> None:
        """
        Register a data consumer with its requirements.
        
        Args:
            requirement: Consumer requirement specification
        """
        self.consumers[requirement.name] = requirement
        logger.info(f"Registered consumer '{requirement.name}'")
    
    def register_producer(
        self,
        name: str,
        schema: Dict[str, str],
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a data producer with its capabilities.
        
        Args:
            name: Producer identifier
            schema: Schema of data produced
            quality_score: Quality score (0.0 to 1.0)
            metadata: Additional metadata
        """
        self.producers[name] = {
            "schema": schema,
            "quality_score": quality_score,
            "metadata": metadata or {}
        }
        logger.info(f"Registered producer '{name}'")
    
    def _calculate_causal_match(
        self,
        producer: str,
        consumer: str,
        consumer_req: ConsumerRequirement
    ) -> float:
        """
        Calculate causal match score between producer and consumer.
        
        Args:
            producer: Producer identifier
            consumer: Consumer identifier
            consumer_req: Consumer requirements
            
        Returns:
            Causal match score (0.0 to 1.0)
        """
        # Check direct causal relationships
        if producer in self.causal_graph:
            if consumer in self.causal_graph[producer]:
                edge_data = self.causal_graph[producer][consumer]
                if isinstance(edge_data, dict):
                    return float(edge_data.get("strength", 0.0))
                return float(edge_data)
        
        # Check if producer satisfies consumer's causal dependencies
        if consumer_req.causal_dependencies:
            if producer in consumer_req.causal_dependencies:
                return 0.8  # High match if producer is in dependencies
        
        # Check indirect paths (producer -> intermediate -> consumer)
        if producer in self.causal_graph:
            for intermediate in self.causal_graph[producer]:
                if intermediate in self.causal_graph:
                    if consumer in self.causal_graph[intermediate]:
                        # Multiply strengths along path
                        path1 = self.causal_graph[producer][intermediate]
                        path2 = self.causal_graph[intermediate][consumer]
                        strength1 = float(path1.get("strength", 0.0) if isinstance(path1, dict) else path1)
                        strength2 = float(path2.get("strength", 0.0) if isinstance(path2, dict) else path2)
                        return strength1 * strength2 * 0.7  # Discount for indirect path
        
        return 0.0
    
    def _calculate_schema_compatibility(
        self,
        producer_schema: Dict[str, str],
        consumer_req: ConsumerRequirement
    ) -> float:
        """
        Calculate schema compatibility score.
        
        Args:
            producer_schema: Schema from producer
            consumer_req: Consumer requirements
            
        Returns:
            Schema compatibility score (0.0 to 1.0)
        """
        if not consumer_req.required_fields:
            return 1.0  # No requirements means full compatibility
        
        producer_fields = set(producer_schema.keys())
        required_fields = set(consumer_req.required_fields)
        
        # Check if all required fields are present
        if not required_fields.issubset(producer_fields):
            missing = required_fields - producer_fields
            logger.debug(f"Missing fields: {missing}")
            return 0.0
        
        # Check type compatibility for required fields
        compatible = 0
        for field in required_fields:
            producer_type = producer_schema.get(field, "")
            preferred_type = consumer_req.schema_preferences.get(field, "")
            
            if preferred_type and producer_type:
                # Simple type matching (can be enhanced)
                if producer_type.lower() == preferred_type.lower():
                    compatible += 1
                elif self._are_types_compatible(producer_type, preferred_type):
                    compatible += 0.8
            else:
                compatible += 1  # No preference means compatible
        
        return compatible / len(required_fields) if required_fields else 1.0
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """
        Check if two types are compatible.
        
        Args:
            type1: First type
            type2: Second type
            
        Returns:
            True if types are compatible
        """
        type_map = {
            "int": ["integer", "int", "number", "numeric"],
            "float": ["float", "double", "number", "numeric", "decimal"],
            "str": ["string", "str", "text", "varchar"],
            "bool": ["boolean", "bool"],
            "datetime": ["timestamp", "date", "datetime"]
        }
        
        for compatible_types in type_map.values():
            if type1.lower() in compatible_types and type2.lower() in compatible_types:
                return True
        return False
    
    def _calculate_quality_match(
        self,
        producer_quality: float,
        consumer_req: ConsumerRequirement
    ) -> float:
        """
        Calculate quality match score.
        
        Args:
            producer_quality: Producer's quality score
            consumer_req: Consumer requirements
            
        Returns:
            Quality match score (0.0 to 1.0)
        """
        if producer_quality >= consumer_req.min_quality_score:
            return 1.0
        elif consumer_req.min_quality_score > 0:
            return producer_quality / consumer_req.min_quality_score
        else:
            return 1.0
    
    def find_routes(
        self,
        consumer_name: Optional[str] = None,
        producer_name: Optional[str] = None
    ) -> List[RouteMatch]:
        """
        Find routes matching producers to consumers.
        
        Args:
            consumer_name: Specific consumer to match (None for all)
            producer_name: Specific producer to match (None for all)
            
        Returns:
            List of route matches sorted by confidence
        """
        matches: List[RouteMatch] = []
        
        consumers_to_match = [consumer_name] if consumer_name else list(self.consumers.keys())
        producers_to_match = [producer_name] if producer_name else list(self.producers.keys())
        
        for consumer in consumers_to_match:
            if consumer not in self.consumers:
                continue
            
            consumer_req = self.consumers[consumer]
            
            for producer in producers_to_match:
                if producer not in self.producers:
                    continue
                
                producer_info = self.producers[producer]
                
                # Calculate individual scores
                causal_score = self._calculate_causal_match(producer, consumer, consumer_req)
                schema_score = self._calculate_schema_compatibility(
                    producer_info["schema"],
                    consumer_req
                )
                quality_score = self._calculate_quality_match(
                    producer_info["quality_score"],
                    consumer_req
                )
                
                # Calculate composite confidence based on strategy
                if self.strategy == RouteMatchStrategy.CAUSAL_ONLY:
                    confidence = causal_score
                elif self.strategy == RouteMatchStrategy.SCHEMA_ONLY:
                    confidence = schema_score
                elif self.strategy == RouteMatchStrategy.QUALITY_ONLY:
                    confidence = quality_score
                else:  # COMPOSITE
                    # Weighted average: causal 40%, schema 40%, quality 20%
                    confidence = (
                        causal_score * 0.4 +
                        schema_score * 0.4 +
                        quality_score * 0.2
                    )
                
                # Only include matches with minimum threshold
                if confidence > 0.1:
                    reasoning = (
                        f"Causal: {causal_score:.2f}, "
                        f"Schema: {schema_score:.2f}, "
                        f"Quality: {quality_score:.2f}"
                    )
                    
                    match = RouteMatch(
                        producer=producer,
                        consumer=consumer,
                        confidence=confidence,
                        causal_strength=causal_score,
                        schema_compatibility=schema_score,
                        quality_score=quality_score,
                        reasoning=reasoning,
                        metadata={
                            "strategy": self.strategy.value,
                            "producer_metadata": producer_info.get("metadata", {})
                        }
                    )
                    matches.append(match)
        
        # Sort by confidence (descending)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches
    
    def get_best_route(
        self,
        consumer_name: str,
        producer_name: Optional[str] = None
    ) -> Optional[RouteMatch]:
        """
        Get the best route for a consumer.
        
        Args:
            consumer_name: Consumer identifier
            producer_name: Optional specific producer to match
            
        Returns:
            Best route match or None if no match found
        """
        matches = self.find_routes(consumer_name=consumer_name, producer_name=producer_name)
        return matches[0] if matches else None
    
    def update_causal_graph(self, causal_graph: Dict[str, Dict[str, float]]) -> None:
        """
        Update the causal graph used for routing.
        
        Args:
            causal_graph: Updated causal graph
        """
        self.causal_graph = causal_graph
        logger.info("Updated causal graph in routing engine")

