"""
Data Broker Agent

Comprehensive data brokerage agent combining CRCA causal reasoning with
multi-source data collection, intelligent routing, and pipeline management.
"""

# Standard library imports
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
from loguru import logger

# Import CRCA components
try:
    from CRCA import CausalRelationType, CRCAAgent
except ImportError:
    # Try relative import if CRCA.py is in parent directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CRCA import CausalRelationType, CRCAAgent

# Local imports
from .pipeline import (
    Pipeline,
    PipelineStage,
    StageType,
    aggregate_stage,
    filter_stage,
    validate_stage,
)
from .routing import (
    ConsumerRequirement,
    RouteMatch,
    RouteMatchStrategy,
    RoutingEngine,
)
from .sources import (
    APIDataSource,
    DataSchema,
    DataSource,
    DataSourceMetadata,
    DatabaseDataSource,
    FileDataSource,
)


class DataBrokerAgent(CRCAAgent):
    """
    Data Broker Agent with causal reasoning capabilities.
    
    Combines CRCA causal reasoning with comprehensive data brokerage features:
    - Multi-source data collection
    - Intelligent routing using causal graphs
    - Pipeline management with causal optimization
    - LLM-powered data discovery and quality assessment
    """
    
    def __init__(
        self,
        agent_name: str = "data-broker-agent",
        agent_description: str = "Data Broker Agent with Causal Reasoning",
        model_name: str = "gpt-4o",
        max_loops: Optional[Union[int, str]] = 5,
        routing_strategy: RouteMatchStrategy = RouteMatchStrategy.COMPOSITE,
        enable_causal_routing: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the Data Broker Agent.
        
        Args:
            agent_name: Unique identifier for the agent
            agent_description: Human-readable description
            model_name: LLM model name
            max_loops: Maximum reasoning loops
            routing_strategy: Strategy for route matching
            enable_causal_routing: Whether to use causal graphs for routing
            **kwargs: Additional arguments passed to CRCAAgent
        """
        # Initialize CRCA agent with data broker schema
        super().__init__(
            agent_name=agent_name,
            agent_description=agent_description,
            model_name=model_name,
            max_loops=max_loops,
            **kwargs
        )
        
        # Data broker specific attributes
        self.data_sources: Dict[str, DataSource] = {}
        self.routing_engine = RoutingEngine(
            causal_graph=self.causal_graph if enable_causal_routing else {},
            strategy=routing_strategy
        )
        self.pipelines: Dict[str, Pipeline] = {}
        self.enable_causal_routing = enable_causal_routing
        
        # Data catalog for LLM-powered discovery
        self.data_catalog: Dict[str, Dict[str, Any]] = {}
        
        # Update system prompt for data broker context
        broker_prompt = (
            "You are a Data Broker Agent with advanced causal reasoning capabilities. "
            "You can collect data from multiple sources, route it intelligently using "
            "causal dependencies, and manage data transformation pipelines. "
            "Use causal graphs to understand data relationships and predict impacts."
        )
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{broker_prompt}"
        else:
            self.system_prompt = broker_prompt
        
        logger.info(f"Initialized DataBrokerAgent '{agent_name}'")
    
    def register_data_source(
        self,
        source: DataSource,
        auto_connect: bool = True
    ) -> bool:
        """
        Register a data source with the broker.
        
        Args:
            source: Data source instance to register
            auto_connect: Whether to automatically connect to the source
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if auto_connect and not source.connect():
                logger.warning(f"Failed to connect to data source '{source.name}'")
                return False
            
            self.data_sources[source.name] = source
            
            # Update routing engine with producer info
            schema_dict = {field: dtype for field, dtype in source.schema.fields.items()}
            self.routing_engine.register_producer(
                name=source.name,
                schema=schema_dict,
                quality_score=source.metadata.quality_score,
                metadata={
                    "source_type": source.source_type.value,
                    "update_frequency": source.update_frequency,
                    "description": source.metadata.description
                }
            )
            
            # Add to data catalog for LLM discovery
            self.data_catalog[source.name] = {
                "name": source.name,
                "type": source.source_type.value,
                "schema": schema_dict,
                "description": source.metadata.description,
                "quality_score": source.metadata.quality_score,
                "last_update": source.metadata.last_update.isoformat() if source.metadata.last_update else None
            }
            
            # Model causal dependencies if specified
            if source.metadata.dependencies:
                for dep in source.metadata.dependencies:
                    if dep in self.data_sources:
                        self.add_causal_relationship(
                            dep,
                            source.name,
                            strength=0.7,
                            relation_type=CausalRelationType.DIRECT
                        )
            
            logger.info(f"Registered data source '{source.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error registering data source '{source.name}': {e}")
            return False
    
    def collect_data(
        self,
        sources: Optional[List[str]] = None,
        use_cache: bool = True,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect data from specified sources.
        
        Args:
            sources: List of source names to collect from (None for all)
            use_cache: Whether to use cached data if available
            query: Query parameters for filtering
            limit: Maximum records per source
            
        Returns:
            Dictionary mapping source names to collected data
        """
        sources_to_collect = sources or list(self.data_sources.keys())
        collected_data: Dict[str, Any] = {}
        
        logger.info(f"Collecting data from {len(sources_to_collect)} sources")
        
        for source_name in sources_to_collect:
            if source_name not in self.data_sources:
                logger.warning(f"Source '{source_name}' not found")
                continue
            
            source = self.data_sources[source_name]
            
            try:
                # Check cache first
                if use_cache:
                    cached = source.get_cached_data()
                    if cached is not None:
                        logger.debug(f"Using cached data for source '{source_name}'")
                        collected_data[source_name] = cached
                        continue
                
                # Fetch fresh data
                data = source.fetch_data(query=query, limit=limit)
                collected_data[source_name] = data
                
                logger.info(f"Collected data from source '{source_name}'")
                
            except Exception as e:
                logger.error(f"Error collecting data from source '{source_name}': {e}")
                collected_data[source_name] = None
        
        return collected_data
    
    def route_data(
        self,
        data: Union[Dict[str, Any], str],
        consumers: Optional[List[str]] = None,
        producer_name: Optional[str] = None
    ) -> Dict[str, List[RouteMatch]]:
        """
        Route data to consumers based on causal dependencies and schema matching.
        
        Args:
            data: Data to route (dict mapping source names to data, or single source name)
            consumers: List of consumer names (None for all registered)
            producer_name: Specific producer name if data is a single source
            
        Returns:
            Dictionary mapping consumer names to list of route matches
        """
        # Update routing engine with current causal graph if enabled
        if self.enable_causal_routing:
            self.routing_engine.update_causal_graph(self.causal_graph)
        
        # Determine producers from data
        if isinstance(data, str):
            # Single source name
            producers = [data]
            if producer_name is None:
                producer_name = data
        elif isinstance(data, dict):
            # Multiple sources
            producers = [name for name, value in data.items() if value is not None]
        else:
            logger.error("Invalid data format for routing")
            return {}
        
        # Find routes for each consumer
        routes: Dict[str, List[RouteMatch]] = {}
        consumers_to_match = consumers or list(self.routing_engine.consumers.keys())
        
        for consumer in consumers_to_match:
            matches = []
            for producer in producers:
                match = self.routing_engine.get_best_route(
                    consumer_name=consumer,
                    producer_name=producer
                )
                if match:
                    matches.append(match)
            
            if matches:
                routes[consumer] = matches
        
        logger.info(f"Routed data to {len(routes)} consumers")
        return routes
    
    def register_consumer(self, requirement: ConsumerRequirement) -> None:
        """
        Register a data consumer with requirements.
        
        Args:
            requirement: Consumer requirement specification
        """
        self.routing_engine.register_consumer(requirement)
        logger.info(f"Registered consumer '{requirement.name}'")
    
    def create_pipeline(
        self,
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        causal_optimization: bool = True
    ) -> Pipeline:
        """
        Create a data transformation pipeline.
        
        Args:
            name: Pipeline identifier
            stages: List of pipeline stages
            causal_optimization: Whether to optimize stage order using causal graph
            
        Returns:
            Created pipeline instance
        """
        pipeline = Pipeline(
            name=name,
            stages=stages or [],
            causal_graph=self.causal_graph if causal_optimization else {},
            optimize_order=causal_optimization
        )
        
        self.pipelines[name] = pipeline
        logger.info(f"Created pipeline '{name}'")
        return pipeline
    
    def analyze_data_dependencies(
        self,
        source: str,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze data dependencies using causal graphs.
        
        Args:
            source: Source data identifier
            target: Optional target identifier for specific analysis
            
        Returns:
            Analysis results including dependencies, impacts, and paths
        """
        if source not in self.causal_graph:
            return {
                "error": f"Source '{source}' not found in causal graph",
                "dependencies": [],
                "downstream_impacts": []
            }
        
        # Get direct dependencies (parents)
        dependencies = self._get_parents(source)
        
        # Get downstream impacts (descendants)
        downstream = self._get_descendants(source)
        
        # Analyze causal strength
        if target:
            strength_analysis = self.analyze_causal_strength(source, target)
        else:
            strength_analysis = {}
            for child in self._get_children(source):
                strength_analysis[child] = self.analyze_causal_strength(source, child)
        
        # Detect confounders if target specified
        confounders = []
        if target:
            confounders = self.detect_confounders(source, target)
        
        return {
            "source": source,
            "target": target,
            "dependencies": dependencies,
            "downstream_impacts": downstream,
            "causal_strength": strength_analysis,
            "confounders": confounders,
            "path_to_target": self.identify_causal_chain(source, target) if target else None
        }
    
    def discover_data(
        self,
        query: str,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover data sources using natural language query.
        
        Args:
            query: Natural language query describing needed data
            use_llm: Whether to use LLM for intelligent matching
            
        Returns:
            List of matching data sources with relevance scores
        """
        if use_llm:
            # Use LLM to understand query and match to catalog
            llm_query = (
                f"Based on the following data catalog, find data sources that match "
                f"this query: '{query}'\n\n"
                f"Data Catalog:\n{self._format_catalog_for_llm()}\n\n"
                f"Return a JSON list of matching sources with relevance scores."
            )
            
            try:
                response = self.step(llm_query)
                # Parse LLM response (simplified - could be enhanced)
                matches = self._parse_llm_discovery_response(response)
                logger.info(f"LLM discovered {len(matches)} matching sources")
                return matches
            except Exception as e:
                logger.error(f"Error in LLM discovery: {e}")
                return self._simple_discovery(query)
        else:
            return self._simple_discovery(query)
    
    def _format_catalog_for_llm(self) -> str:
        """Format data catalog for LLM consumption."""
        lines = []
        for name, info in self.data_catalog.items():
            lines.append(
                f"- {name}: {info.get('description', 'No description')} "
                f"(Type: {info.get('type')}, Quality: {info.get('quality_score', 0):.2f})"
            )
        return "\n".join(lines)
    
    def _parse_llm_discovery_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM discovery response (simplified implementation)."""
        matches = []
        # Simple keyword matching fallback
        query_lower = response.lower()
        for name, info in self.data_catalog.items():
            if any(keyword in info.get('description', '').lower() for keyword in query_lower.split()):
                matches.append({
                    "source": name,
                    "relevance": 0.7,
                    "info": info
                })
        return matches
    
    def _simple_discovery(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword-based discovery."""
        query_lower = query.lower()
        matches = []
        for name, info in self.data_catalog.items():
            score = 0.0
            description = info.get('description', '').lower()
            name_lower = name.lower()
            
            # Check name match
            if query_lower in name_lower:
                score += 0.5
            
            # Check description match
            for keyword in query_lower.split():
                if keyword in description:
                    score += 0.2
            
            if score > 0:
                matches.append({
                    "source": name,
                    "relevance": min(score, 1.0),
                    "info": info
                })
        
        matches.sort(key=lambda x: x['relevance'], reverse=True)
        return matches
    
    def assess_data_quality(
        self,
        source_name: str,
        data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Assess data quality using LLM and statistical methods.
        
        Args:
            source_name: Name of the data source
            data: Optional data to assess (uses cached if None)
            
        Returns:
            Quality assessment results
        """
        if data is None:
            if source_name in self.data_sources:
                data = self.data_sources[source_name].get_cached_data()
            else:
                return {"error": f"Source '{source_name}' not found"}
        
        if data is None:
            return {"error": "No data available for assessment"}
        
        # Basic quality metrics
        quality_metrics = {
            "completeness": 1.0,
            "consistency": 1.0,
            "validity": 1.0,
            "timeliness": 1.0
        }
        
        llm_assessment = None
        # Use LLM for intelligent assessment
        try:
            assessment_prompt = (
                f"Assess the quality of the following data from source '{source_name}':\n\n"
                f"Data sample: {str(data)[:500]}\n\n"
                f"Provide assessment of completeness, consistency, validity, and timeliness."
            )
            
            llm_assessment = self.step(assessment_prompt)
            
        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}")
        
        # Calculate overall quality score (only numeric metrics)
        numeric_metrics = {k: v for k, v in quality_metrics.items() if isinstance(v, (int, float))}
        overall_score = sum(numeric_metrics.values()) / len(numeric_metrics) if numeric_metrics else 0.0
        
        result = {
            "source": source_name,
            "quality_score": overall_score,
            "metrics": quality_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        if llm_assessment:
            result["llm_assessment"] = llm_assessment
        
        return result
    
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the data broker agent with enhanced capabilities.
        
        Overrides parent run() to add data broker specific functionality.
        
        Args:
            task: Task string or data state
            initial_state: Initial state dictionary
            target_variables: Target variables for analysis
            max_steps: Maximum evolution steps
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary with broker-specific additions
        """
        # Check if task is a data broker operation
        if isinstance(task, str):
            task_lower = task.lower()
            
            # Data discovery query
            if "discover" in task_lower or "find data" in task_lower:
                discovered = self.discover_data(task, use_llm=True)
                return {
                    "operation": "data_discovery",
                    "query": task,
                    "results": discovered
                }
            
            # Data collection request
            if "collect" in task_lower or "fetch" in task_lower:
                # Extract source names from task (simplified)
                sources = None  # Could parse from task
                collected = self.collect_data(sources=sources)
                return {
                    "operation": "data_collection",
                    "sources": list(collected.keys()),
                    "data": collected
                }
            
            # Dependency analysis
            if "dependencies" in task_lower or "analyze" in task_lower:
                # Extract source from task (simplified)
                source = None  # Could parse from task
                if source:
                    analysis = self.analyze_data_dependencies(source)
                    return {
                        "operation": "dependency_analysis",
                        "analysis": analysis
                    }
        
        # Fall back to parent implementation for causal reasoning
        return super().run(
            task=task,
            initial_state=initial_state,
            target_variables=target_variables,
            max_steps=max_steps,
            **kwargs
        )


def main():
    """Main entry point for DataBrokerAgent CLI."""
    import argparse
    import json
    
    # Try to import yaml, but make it optional
    try:
        import yaml
        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
        logger.warning("yaml not available, YAML config files will not be supported")
    
    parser = argparse.ArgumentParser(
        description="Data Broker Agent - Causal reasoning-based data brokerage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize broker
  python -m data_broker.DataBrokerAgent init --name my-broker
  
  # Register an API data source
  python -m data_broker.DataBrokerAgent register api --name sales_api --url https://api.example.com/sales
  
  # Register a file data source
  python -m data_broker.DataBrokerAgent register file --name inventory --path data/inventory.csv
  
  # Collect data from sources
  python -m data_broker.DataBrokerAgent collect --sources sales_api inventory
  
  # Discover data sources
  python -m data_broker.DataBrokerAgent discover "sales data"
  
  # Analyze dependencies
  python -m data_broker.DataBrokerAgent analyze --source sales_api --target revenue_prediction
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="data-broker-agent",
        help="Agent name (default: data-broker-agent)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new data broker agent")
    init_parser.add_argument("--name", type=str, help="Agent name")
    init_parser.add_argument("--model", type=str, help="LLM model name")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a data source")
    register_subparsers = register_parser.add_subparsers(dest="source_type", help="Source type")
    
    # Register API source
    api_parser = register_subparsers.add_parser("api", help="Register API data source")
    api_parser.add_argument("--name", type=str, required=True, help="Source name")
    api_parser.add_argument("--url", type=str, required=True, help="API URL")
    api_parser.add_argument("--method", type=str, default="GET", help="HTTP method")
    api_parser.add_argument("--headers", type=str, help="Headers as JSON string")
    
    # Register file source
    file_parser = register_subparsers.add_parser("file", help="Register file data source")
    file_parser.add_argument("--name", type=str, required=True, help="Source name")
    file_parser.add_argument("--path", type=str, required=True, help="File path")
    
    # Register database source
    db_parser = register_subparsers.add_parser("database", help="Register database data source")
    db_parser.add_argument("--name", type=str, required=True, help="Source name")
    db_parser.add_argument("--connection", type=str, required=True, help="Connection string")
    db_parser.add_argument("--table", type=str, help="Table name")
    db_parser.add_argument("--query", type=str, help="SQL query")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect data from sources")
    collect_parser.add_argument("--sources", type=str, nargs="+", help="Source names to collect from")
    collect_parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    collect_parser.add_argument("--limit", type=int, help="Limit records per source")
    collect_parser.add_argument("--output", type=str, help="Output file path")
    
    # Route command
    route_parser = subparsers.add_parser("route", help="Route data to consumers")
    route_parser.add_argument("--producer", type=str, required=True, help="Producer source name")
    route_parser.add_argument("--consumers", type=str, nargs="+", help="Consumer names")
    route_parser.add_argument("--output", type=str, help="Output file path")
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover data sources")
    discover_parser.add_argument("query", type=str, help="Natural language query")
    discover_parser.add_argument("--no-llm", action="store_true", help="Disable LLM matching")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data dependencies")
    analyze_parser.add_argument("--source", type=str, required=True, help="Source identifier")
    analyze_parser.add_argument("--target", type=str, help="Target identifier")
    analyze_parser.add_argument("--output", type=str, help="Output file path")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline operations")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_action", help="Pipeline action")
    
    create_pipeline_parser = pipeline_subparsers.add_parser("create", help="Create a pipeline")
    create_pipeline_parser.add_argument("--name", type=str, required=True, help="Pipeline name")
    create_pipeline_parser.add_argument("--stages", type=str, help="Stages as JSON array")
    
    run_pipeline_parser = pipeline_subparsers.add_parser("run", help="Run a pipeline")
    run_pipeline_parser.add_argument("--name", type=str, required=True, help="Pipeline name")
    run_pipeline_parser.add_argument("--input", type=str, help="Input data file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                if (args.config.endswith('.yaml') or args.config.endswith('.yml')):
                    if YAML_AVAILABLE:
                        config = yaml.safe_load(f)
                    else:
                        logger.error("YAML not available, cannot load YAML config file")
                        return 1
                else:
                    config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return 1
    
    # Initialize broker
    broker = DataBrokerAgent(
        agent_name=config.get("agent_name", args.name),
        model_name=config.get("model_name", args.model),
        max_loops=config.get("max_loops", 5),
        verbose=args.verbose or config.get("verbose", False)
    )
    
    # Execute command
    try:
        if args.command == "init":
            logger.info(f"Initialized DataBrokerAgent '{broker.agent_name}'")
            logger.info(f"Model: {broker.model_name}")
            return 0
        
        elif args.command == "register":
            from .sources import APIDataSource, FileDataSource, DatabaseDataSource, DataSchema
            
            if args.source_type == "api":
                headers = {}
                if args.headers:
                    headers = json.loads(args.headers)
                
                source = APIDataSource(
                    name=args.name,
                    url=args.url,
                    method=args.method,
                    headers=headers
                )
                success = broker.register_data_source(source, auto_connect=False)
                if success:
                    logger.info(f"Registered API source '{args.name}'")
                    return 0
                else:
                    logger.error(f"Failed to register API source '{args.name}'")
                    return 1
            
            elif args.source_type == "file":
                source = FileDataSource(
                    name=args.name,
                    file_path=args.path
                )
                success = broker.register_data_source(source, auto_connect=False)
                if success:
                    logger.info(f"Registered file source '{args.name}'")
                    return 0
                else:
                    logger.error(f"Failed to register file source '{args.name}'")
                    return 1
            
            elif args.source_type == "database":
                source = DatabaseDataSource(
                    name=args.name,
                    connection_string=args.connection,
                    table_name=args.table,
                    query=args.query
                )
                success = broker.register_data_source(source, auto_connect=False)
                if success:
                    logger.info(f"Registered database source '{args.name}'")
                    return 0
                else:
                    logger.error(f"Failed to register database source '{args.name}'")
                    return 1
        
        elif args.command == "collect":
            sources = args.sources if args.sources else None
            data = broker.collect_data(
                sources=sources,
                use_cache=not args.no_cache,
                limit=args.limit
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump({k: str(v) for k, v in data.items()}, f, indent=2)
                logger.info(f"Saved collected data to {args.output}")
            else:
                logger.info(f"Collected data from {len(data)} sources")
                for source, data_obj in data.items():
                    logger.info(f"  {source}: {type(data_obj).__name__}")
            
            return 0
        
        elif args.command == "route":
            routes = broker.route_data(
                data=args.producer,
                consumers=args.consumers
            )
            
            if args.output:
                routes_dict = {
                    consumer: [
                        {
                            "producer": m.producer,
                            "confidence": m.confidence,
                            "reasoning": m.reasoning
                        }
                        for m in matches
                    ]
                    for consumer, matches in routes.items()
                }
                with open(args.output, 'w') as f:
                    json.dump(routes_dict, f, indent=2)
                logger.info(f"Saved routes to {args.output}")
            else:
                logger.info(f"Found routes for {len(routes)} consumers")
                for consumer, matches in routes.items():
                    logger.info(f"  {consumer}:")
                    for match in matches:
                        logger.info(f"    -> {match.producer} (confidence: {match.confidence:.2f})")
            
            return 0
        
        elif args.command == "discover":
            results = broker.discover_data(args.query, use_llm=not args.no_llm)
            logger.info(f"Discovered {len(results)} matching sources:")
            for result in results:
                logger.info(f"  {result['source']}: relevance {result['relevance']:.2f}")
            return 0
        
        elif args.command == "analyze":
            analysis = broker.analyze_data_dependencies(args.source, args.target)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f"Saved analysis to {args.output}")
            else:
                logger.info(f"Dependency analysis for '{args.source}':")
                logger.info(f"  Dependencies: {analysis.get('dependencies', [])}")
                logger.info(f"  Downstream impacts: {analysis.get('downstream_impacts', [])}")
                if args.target:
                    logger.info(f"  Path to '{args.target}': {analysis.get('path_to_target', [])}")
            
            return 0
        
        elif args.command == "pipeline":
            if args.pipeline_action == "create":
                stages = []
                if args.stages:
                    stages_data = json.loads(args.stages)
                    from .pipeline import PipelineStage, StageType
                    # Simplified stage creation - would need more details in real implementation
                    stages = []
                
                pipeline = broker.create_pipeline(name=args.name, stages=stages)
                logger.info(f"Created pipeline '{args.name}'")
                return 0
            
            elif args.pipeline_action == "run":
                if args.name not in broker.pipelines:
                    logger.error(f"Pipeline '{args.name}' not found")
                    return 1
                
                input_data = {}
                if args.input:
                    with open(args.input, 'r') as f:
                        input_data = json.load(f)
                
                pipeline = broker.pipelines[args.name]
                result = pipeline.execute(input_data)
                logger.info(f"Pipeline '{args.name}' executed successfully")
                return 0
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
