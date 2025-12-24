"""
Pipeline Framework

Provides data transformation pipelines with causal optimization.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class StageType(Enum):
    """Types of pipeline stages."""
    FILTER = "filter"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    JOIN = "join"
    CUSTOM = "custom"


@dataclass
class PipelineStage:
    """Represents a stage in a data pipeline."""
    
    name: str
    stage_type: StageType
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class Pipeline:
    """Data transformation pipeline with causal optimization."""
    
    def __init__(
        self,
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        causal_graph: Optional[Dict[str, Dict[str, float]]] = None,
        optimize_order: bool = True
    ) -> None:
        """
        Initialize a pipeline.
        
        Args:
            name: Pipeline identifier
            stages: List of pipeline stages
            causal_graph: Causal graph for optimization
            optimize_order: Whether to optimize stage order using causal graph
        """
        self.name = name
        self.stages: List[PipelineStage] = stages or []
        self.causal_graph = causal_graph or {}
        self.optimize_order = optimize_order
        self.execution_history: List[Dict[str, Any]] = []
        
        if self.optimize_order and self.causal_graph:
            self._optimize_stage_order()
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            stage: Pipeline stage to add
        """
        self.stages.append(stage)
        if self.optimize_order and self.causal_graph:
            self._optimize_stage_order()
        logger.info(f"Added stage '{stage.name}' to pipeline '{self.name}'")
    
    def _optimize_stage_order(self) -> None:
        """Optimize stage order based on causal dependencies."""
        if not self.stages:
            return
        
        # Build dependency graph
        stage_deps: Dict[str, List[str]] = {}
        stage_map: Dict[str, PipelineStage] = {}
        
        for stage in self.stages:
            stage_map[stage.name] = stage
            stage_deps[stage.name] = stage.dependencies.copy()
            
            # Add causal dependencies from causal graph
            if stage.name in self.causal_graph:
                for dependent in self.causal_graph[stage.name]:
                    if dependent not in stage_deps[stage.name]:
                        stage_deps[stage.name].append(dependent)
        
        # Topological sort to determine optimal order
        ordered: List[str] = []
        visited: set = set()
        temp_visited: set = set()
        
        def visit(stage_name: str) -> None:
            if stage_name in temp_visited:
                logger.warning(f"Circular dependency detected involving '{stage_name}'")
                return
            if stage_name in visited:
                return
            
            temp_visited.add(stage_name)
            for dep in stage_deps.get(stage_name, []):
                if dep in stage_map:
                    visit(dep)
            temp_visited.remove(stage_name)
            visited.add(stage_name)
            ordered.append(stage_name)
        
        for stage in self.stages:
            if stage.name not in visited:
                visit(stage.name)
        
        # Reorder stages
        ordered_stages = [stage_map[name] for name in ordered if name in stage_map]
        self.stages = ordered_stages
        logger.info(f"Optimized pipeline '{self.name}' stage order: {[s.name for s in self.stages]}")
    
    def execute(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute the pipeline on data.
        
        Args:
            data: Input data
            context: Execution context
            
        Returns:
            Transformed data
        """
        context = context or {}
        current_data = data
        
        logger.info(f"Executing pipeline '{self.name}' with {len(self.stages)} stages")
        
        for i, stage in enumerate(self.stages):
            try:
                logger.debug(f"Executing stage {i+1}/{len(self.stages)}: '{stage.name}'")
                
                # Prepare stage context
                stage_context = {
                    **context,
                    "stage_index": i,
                    "stage_name": stage.name,
                    "pipeline_name": self.name
                }
                
                # Execute stage
                if isinstance(current_data, dict) and "data" in current_data:
                    # Handle wrapped data
                    result = stage.function(current_data["data"], stage_context, **stage.config)
                    current_data["data"] = result
                else:
                    current_data = stage.function(current_data, stage_context, **stage.config)
                
                # Record execution
                self.execution_history.append({
                    "stage": stage.name,
                    "stage_type": stage.stage_type.value,
                    "success": True,
                    "input_shape": self._get_data_shape(current_data),
                    "output_shape": self._get_data_shape(current_data)
                })
                
            except Exception as e:
                logger.error(f"Error in stage '{stage.name}': {e}")
                self.execution_history.append({
                    "stage": stage.name,
                    "stage_type": stage.stage_type.value,
                    "success": False,
                    "error": str(e)
                })
                raise
        
        logger.info(f"Pipeline '{self.name}' completed successfully")
        return current_data
    
    def _get_data_shape(self, data: Any) -> str:
        """
        Get shape description of data.
        
        Args:
            data: Data to describe
            
        Returns:
            Shape description string
        """
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return f"DataFrame({data.shape[0]} rows, {data.shape[1]} cols)"
        elif isinstance(data, list):
            return f"List({len(data)} items)"
        elif isinstance(data, dict):
            return f"Dict({len(data)} keys)"
        else:
            return str(type(data).__name__)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline execution.
        
        Returns:
            Execution summary dictionary
        """
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("success", False))
        
        return {
            "pipeline_name": self.name,
            "total_stages": total,
            "successful_stages": successful,
            "failed_stages": total - successful,
            "stages": self.execution_history
        }
    
    def reset(self) -> None:
        """Reset pipeline execution history."""
        self.execution_history = []
        logger.info(f"Reset pipeline '{self.name}' execution history")


# Common pipeline stage functions

def filter_stage(
    data: Any,
    context: Dict[str, Any],
    condition: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Filter data based on condition.
    
    Args:
        data: Data to filter
        context: Execution context
        condition: Filter condition function
        **kwargs: Additional parameters
        
    Returns:
        Filtered data
    """
    if condition is None:
        return data
    
    if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        return data[data.apply(condition, axis=1)]
    elif isinstance(data, list):
        return [item for item in data if condition(item)]
    else:
        logger.warning("Filter stage not supported for this data type")
        return data


def validate_stage(
    data: Any,
    context: Dict[str, Any],
    schema: Optional[Dict[str, str]] = None,
    required_fields: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Validate data against schema.
    
    Args:
        data: Data to validate
        context: Execution context
        schema: Expected schema
        required_fields: Required field names
        **kwargs: Additional parameters
        
    Returns:
        Validated data (raises exception if invalid)
    """
    if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        if required_fields:
            missing = set(required_fields) - set(data.columns)
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
        return data
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        if required_fields:
            missing = set(required_fields) - set(data[0].keys())
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
        return data
    else:
        logger.warning("Validation stage not fully supported for this data type")
        return data


def aggregate_stage(
    data: Any,
    context: Dict[str, Any],
    group_by: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, str]] = None,
    **kwargs
) -> Any:
    """
    Aggregate data.
    
    Args:
        data: Data to aggregate
        context: Execution context
        group_by: Fields to group by
        aggregations: Aggregation functions
        **kwargs: Additional parameters
        
    Returns:
        Aggregated data
    """
    if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        if group_by and aggregations:
            return data.groupby(group_by).agg(aggregations).reset_index()
        elif group_by:
            return data.groupby(group_by).mean().reset_index()
        else:
            return data.agg(aggregations) if aggregations else data
    else:
        logger.warning("Aggregation stage requires pandas DataFrame")
        return data

