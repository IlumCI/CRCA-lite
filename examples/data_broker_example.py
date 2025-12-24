"""
Example: Data Broker Agent Usage

Demonstrates comprehensive data broker capabilities including:
- Multi-source data collection
- Causal dependency modeling
- Intelligent data routing
- Pipeline management
- LLM-powered data discovery
"""

from typing import Dict, Any, List
from loguru import logger

# Import data broker components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_broker import (
    DataBrokerAgent,
    APIDataSource,
    DatabaseDataSource,
    FileDataSource,
    DataSchema,
    ConsumerRequirement,
    PipelineStage,
    StageType,
    RouteMatchStrategy
)
from data_broker.pipeline import filter_stage, validate_stage, aggregate_stage


def example_basic_setup():
    """Example: Basic data broker setup and data source registration."""
    logger.info("=== Example 1: Basic Setup ===")
    
    # Initialize broker
    broker = DataBrokerAgent(
        agent_name="example-broker",
        model_name="gpt-4o-mini",  # Use cheaper model for examples
        max_loops=3,
        routing_strategy=RouteMatchStrategy.COMPOSITE
    )
    
    # Create and register API data source
    sales_api = APIDataSource(
        name="sales_api",
        url="https://api.example.com/sales",
        method="GET",
        headers={"Authorization": "Bearer token"},
        schema=DataSchema(
            fields={
                "date": "datetime",
                "product_id": "str",
                "quantity": "int",
                "revenue": "float"
            },
            timestamp_field="date"
        ),
        update_frequency=3600.0  # 1 hour
    )
    
    broker.register_data_source(sales_api, auto_connect=False)  # Don't actually connect in example
    
    # Create and register file data source
    inventory_file = FileDataSource(
        name="inventory_file",
        file_path="data/inventory.csv",
        schema=DataSchema(
            fields={
                "product_id": "str",
                "stock_level": "int",
                "warehouse": "str"
            },
            primary_key="product_id"
        )
    )
    
    broker.register_data_source(inventory_file, auto_connect=False)
    
    logger.info(f"Registered {len(broker.data_sources)} data sources")
    return broker


def example_causal_modeling(broker: DataBrokerAgent):
    """Example: Model data dependencies using causal graphs."""
    logger.info("=== Example 2: Causal Dependency Modeling ===")
    
    # Model causal relationships between data sources
    # Sales data affects revenue prediction
    broker.add_causal_relationship(
        "sales_api",
        "revenue_prediction",
        strength=0.8,
        relation_type="direct"
    )
    
    # Inventory data affects revenue prediction (less direct)
    broker.add_causal_relationship(
        "inventory_file",
        "revenue_prediction",
        strength=0.6,
        relation_type="direct"
    )
    
    # Sales data affects inventory levels (feedback loop)
    broker.add_causal_relationship(
        "sales_api",
        "inventory_file",
        strength=0.7,
        relation_type="direct"
    )
    
    # Analyze dependencies
    analysis = broker.analyze_data_dependencies("sales_api", "revenue_prediction")
    logger.info(f"Dependency analysis: {analysis}")
    
    return broker


def example_data_collection(broker: DataBrokerAgent):
    """Example: Collect data from multiple sources."""
    logger.info("=== Example 3: Data Collection ===")
    
    # Collect data from all registered sources
    # Note: This will fail in example since sources aren't actually connected
    try:
        data = broker.collect_data(
            sources=["sales_api", "inventory_file"],
            use_cache=True,
            limit=100
        )
        logger.info(f"Collected data from {len(data)} sources")
        return data
    except Exception as e:
        logger.warning(f"Data collection failed (expected in example): {e}")
        return {}


def example_intelligent_routing(broker: DataBrokerAgent):
    """Example: Intelligent data routing using causal graphs."""
    logger.info("=== Example 4: Intelligent Routing ===")
    
    # Register a consumer with requirements
    analytics_consumer = ConsumerRequirement(
        name="analytics_service",
        required_fields=["product_id", "revenue", "quantity"],
        min_quality_score=0.7,
        causal_dependencies=["sales_api"],
        schema_preferences={
            "product_id": "str",
            "revenue": "float"
        },
        description="Analytics service requiring sales data"
    )
    
    broker.register_consumer(analytics_consumer)
    
    # Find routes
    routes = broker.route_data(
        data="sales_api",
        consumers=["analytics_service"]
    )
    
    for consumer, matches in routes.items():
        logger.info(f"Consumer '{consumer}' matched with:")
        for match in matches:
            logger.info(
                f"  - Producer: {match.producer}, "
                f"Confidence: {match.confidence:.2f}, "
                f"Reasoning: {match.reasoning}"
            )
    
    return routes


def example_pipeline_management(broker: DataBrokerAgent):
    """Example: Create and execute data transformation pipelines."""
    logger.info("=== Example 5: Pipeline Management ===")
    
    # Create pipeline stages
    validate_stage_obj = PipelineStage(
        name="validate",
        stage_type=StageType.VALIDATE,
        function=validate_stage,
        config={
            "required_fields": ["product_id", "quantity", "revenue"]
        },
        description="Validate data schema"
    )
    
    filter_stage_obj = PipelineStage(
        name="filter_positive",
        stage_type=StageType.FILTER,
        function=filter_stage,
        config={
            "condition": lambda row: row.get("quantity", 0) > 0
        },
        dependencies=["validate"],
        description="Filter out zero quantity records"
    )
    
    aggregate_stage_obj = PipelineStage(
        name="aggregate_by_product",
        stage_type=StageType.AGGREGATE,
        function=aggregate_stage,
        config={
            "group_by": ["product_id"],
            "aggregations": {"revenue": "sum", "quantity": "sum"}
        },
        dependencies=["filter_positive"],
        description="Aggregate by product"
    )
    
    # Create pipeline with causal optimization
    pipeline = broker.create_pipeline(
        name="sales_processing",
        stages=[validate_stage_obj, filter_stage_obj, aggregate_stage_obj],
        causal_optimization=True
    )
    
    logger.info(f"Created pipeline '{pipeline.name}' with {len(pipeline.stages)} stages")
    logger.info(f"Stage order: {[s.name for s in pipeline.stages]}")
    
    return pipeline


def example_llm_discovery(broker: DataBrokerAgent):
    """Example: LLM-powered data discovery."""
    logger.info("=== Example 6: LLM-Powered Data Discovery ===")
    
    # Discover data using natural language
    query = "Find data sources related to sales and revenue"
    discovered = broker.discover_data(query, use_llm=True)
    
    logger.info(f"Discovered {len(discovered)} sources for query: '{query}'")
    for match in discovered:
        logger.info(
            f"  - {match['source']}: relevance {match['relevance']:.2f}"
        )
    
    return discovered


def example_comprehensive_workflow():
    """Example: Comprehensive data broker workflow."""
    logger.info("=== Example 7: Comprehensive Workflow ===")
    
    # Setup
    broker = example_basic_setup()
    
    # Model dependencies
    broker = example_causal_modeling(broker)
    
    # Register consumers
    example_intelligent_routing(broker)
    
    # Create pipelines
    pipeline = example_pipeline_management(broker)
    
    # LLM discovery
    example_llm_discovery(broker)
    
    # Run a natural language task
    result = broker.run(
        task="Analyze the causal relationships between sales data and revenue predictions"
    )
    
    logger.info("Comprehensive workflow completed")
    return broker, result


if __name__ == "__main__":
    logger.info("Data Broker Agent Examples")
    logger.info("=" * 50)
    
    # Run examples
    try:
        broker = example_basic_setup()
        broker = example_causal_modeling(broker)
        example_data_collection(broker)
        example_intelligent_routing(broker)
        example_pipeline_management(broker)
        example_llm_discovery(broker)
        
        logger.info("\n" + "=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

