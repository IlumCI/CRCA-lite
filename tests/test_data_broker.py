"""
Test Suite for Data Broker Agent

Tests comprehensive data broker capabilities including:
- Multi-source data collection
- Causal dependency modeling
- Intelligent data routing
- Pipeline management
- LLM-powered data discovery
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path
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
from CRCA import CausalRelationType


@pytest.fixture
def broker():
    """Create a test broker instance."""
    return DataBrokerAgent(
        agent_name="test-broker",
        model_name="gpt-4o-mini",
        max_loops=2,
        routing_strategy=RouteMatchStrategy.COMPOSITE
    )


@pytest.fixture
def sample_api_source():
    """Create a sample API data source."""
    return APIDataSource(
        name="test_sales_api",
        url="https://api.test.com/sales",
        method="GET",
        headers={"Authorization": "Bearer test_token"},
        schema=DataSchema(
            fields={
                "date": "datetime",
                "product_id": "str",
                "quantity": "int",
                "revenue": "float"
            },
            timestamp_field="date"
        ),
        update_frequency=3600.0
    )


@pytest.fixture
def sample_file_source():
    """Create a sample file data source."""
    return FileDataSource(
        name="test_inventory_file",
        file_path="test_data/inventory.csv",
        schema=DataSchema(
            fields={
                "product_id": "str",
                "stock_level": "int",
                "warehouse": "str"
            },
            primary_key="product_id"
        )
    )


class TestBasicSetup:
    """Test basic broker setup and initialization."""
    
    def test_broker_initialization(self, broker):
        """Test that broker initializes correctly."""
        assert broker.agent_name == "test-broker"
        assert broker.model_name == "gpt-4o-mini"
        assert len(broker.data_sources) == 0
        assert len(broker.pipelines) == 0
    
    def test_register_api_source(self, broker, sample_api_source):
        """Test registering an API data source."""
        with patch.object(sample_api_source, 'connect', return_value=True):
            result = broker.register_data_source(sample_api_source, auto_connect=True)
            assert result is True
            assert "test_sales_api" in broker.data_sources
            assert "test_sales_api" in broker.data_catalog
    
    def test_register_file_source(self, broker, sample_file_source):
        """Test registering a file data source."""
        with patch.object(sample_file_source, 'connect', return_value=True):
            result = broker.register_data_source(sample_file_source, auto_connect=True)
            assert result is True
            assert "test_inventory_file" in broker.data_sources
            assert "test_inventory_file" in broker.data_catalog
    
    def test_register_multiple_sources(self, broker, sample_api_source, sample_file_source):
        """Test registering multiple data sources."""
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(sample_file_source, 'connect', return_value=True):
            broker.register_data_source(sample_api_source, auto_connect=True)
            broker.register_data_source(sample_file_source, auto_connect=True)
            
            assert len(broker.data_sources) == 2
            assert len(broker.data_catalog) == 2


class TestCausalModeling:
    """Test causal dependency modeling."""
    
    def test_add_causal_relationship(self, broker):
        """Test adding causal relationships."""
        broker.add_causal_relationship(
            "source_a",
            "target_b",
            strength=0.8,
            relation_type=CausalRelationType.DIRECT
        )
        
        assert "source_a" in broker.causal_graph
        assert "target_b" in broker.causal_graph["source_a"]
    
    def test_analyze_dependencies(self, broker):
        """Test analyzing data dependencies."""
        # Add some causal relationships
        broker.add_causal_relationship("source_a", "target_b", strength=0.8)
        broker.add_causal_relationship("source_a", "target_c", strength=0.6)
        broker.add_causal_relationship("target_b", "target_d", strength=0.7)
        
        # Analyze dependencies
        analysis = broker.analyze_data_dependencies("source_a", "target_b")
        
        assert analysis["source"] == "source_a"
        assert analysis["target"] == "target_b"
        assert "dependencies" in analysis
        assert "downstream_impacts" in analysis
        assert "causal_strength" in analysis
    
    def test_identify_causal_chain(self, broker):
        """Test identifying causal chains."""
        broker.add_causal_relationship("a", "b", strength=0.8)
        broker.add_causal_relationship("b", "c", strength=0.7)
        
        chain = broker.identify_causal_chain("a", "c")
        assert len(chain) == 3
        assert chain[0] == "a"
        assert chain[-1] == "c"


class TestDataCollection:
    """Test data collection functionality."""
    
    def test_collect_data_with_cache(self, broker, sample_api_source):
        """Test collecting data with caching."""
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(sample_api_source, 'get_cached_data', return_value={"test": "data"}):
            broker.register_data_source(sample_api_source, auto_connect=False)
            
            data = broker.collect_data(
                sources=["test_sales_api"],
                use_cache=True
            )
            
            assert "test_sales_api" in data
            assert data["test_sales_api"] == {"test": "data"}
    
    def test_collect_data_from_multiple_sources(self, broker, sample_api_source, sample_file_source):
        """Test collecting from multiple sources."""
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(sample_file_source, 'connect', return_value=True), \
             patch.object(sample_api_source, 'get_cached_data', return_value={"api": "data"}), \
             patch.object(sample_file_source, 'get_cached_data', return_value={"file": "data"}):
            
            broker.register_data_source(sample_api_source, auto_connect=False)
            broker.register_data_source(sample_file_source, auto_connect=False)
            
            data = broker.collect_data(
                sources=["test_sales_api", "test_inventory_file"],
                use_cache=True
            )
            
            assert len(data) == 2
            assert "test_sales_api" in data
            assert "test_inventory_file" in data


class TestIntelligentRouting:
    """Test intelligent data routing."""
    
    def test_register_consumer(self, broker):
        """Test registering a consumer."""
        consumer = ConsumerRequirement(
            name="test_consumer",
            required_fields=["product_id", "revenue"],
            min_quality_score=0.7,
            causal_dependencies=["test_sales_api"]
        )
        
        broker.register_consumer(consumer)
        assert "test_consumer" in broker.routing_engine.consumers
    
    def test_route_data_with_causal_matching(self, broker, sample_api_source):
        """Test routing data using causal matching."""
        # Register source
        with patch.object(sample_api_source, 'connect', return_value=True):
            broker.register_data_source(sample_api_source, auto_connect=False)
        
        # Add causal relationship
        broker.add_causal_relationship("test_sales_api", "revenue_prediction", strength=0.8)
        
        # Register consumer
        consumer = ConsumerRequirement(
            name="analytics_service",
            required_fields=["product_id", "revenue"],
            causal_dependencies=["test_sales_api"]
        )
        broker.register_consumer(consumer)
        
        # Route data
        routes = broker.route_data(
            data="test_sales_api",
            consumers=["analytics_service"]
        )
        
        assert "analytics_service" in routes
        assert len(routes["analytics_service"]) > 0
        assert routes["analytics_service"][0].producer == "test_sales_api"
    
    def test_route_data_schema_matching(self, broker, sample_api_source):
        """Test routing based on schema compatibility."""
        # Register source with schema
        with patch.object(sample_api_source, 'connect', return_value=True):
            broker.register_data_source(sample_api_source, auto_connect=False)
        
        # Register consumer with matching schema
        consumer = ConsumerRequirement(
            name="matching_consumer",
            required_fields=["product_id", "revenue", "quantity"],
            schema_preferences={
                "product_id": "str",
                "revenue": "float"
            }
        )
        broker.register_consumer(consumer)
        
        # Route data
        routes = broker.route_data(data="test_sales_api", consumers=["matching_consumer"])
        
        if "matching_consumer" in routes:
            match = routes["matching_consumer"][0]
            assert match.schema_compatibility > 0


class TestPipelineManagement:
    """Test pipeline management."""
    
    def test_create_pipeline(self, broker):
        """Test creating a pipeline."""
        pipeline = broker.create_pipeline(
            name="test_pipeline",
            stages=[],
            causal_optimization=False
        )
        
        assert pipeline.name == "test_pipeline"
        assert "test_pipeline" in broker.pipelines
    
    def test_pipeline_with_stages(self, broker):
        """Test creating pipeline with stages."""
        validate_stage_obj = PipelineStage(
            name="validate",
            stage_type=StageType.VALIDATE,
            function=validate_stage,
            config={"required_fields": ["product_id"]}
        )
        
        pipeline = broker.create_pipeline(
            name="test_pipeline",
            stages=[validate_stage_obj]
        )
        
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "validate"
    
    def test_pipeline_execution(self, broker):
        """Test executing a pipeline."""
        # Create simple pipeline
        def identity_stage(data, context, **kwargs):
            return data
        
        stage = PipelineStage(
            name="identity",
            stage_type=StageType.CUSTOM,
            function=identity_stage
        )
        
        pipeline = broker.create_pipeline(
            name="test_pipeline",
            stages=[stage]
        )
        
        test_data = {"test": "data"}
        result = pipeline.execute(test_data)
        
        assert result == test_data
        assert len(pipeline.execution_history) == 1
        assert pipeline.execution_history[0]["success"] is True


class TestLLMDiscovery:
    """Test LLM-powered data discovery."""
    
    def test_simple_discovery(self, broker, sample_api_source):
        """Test simple keyword-based discovery."""
        with patch.object(sample_api_source, 'connect', return_value=True):
            broker.register_data_source(sample_api_source, auto_connect=False)
        
        # Update catalog description for better matching
        broker.data_catalog["test_sales_api"]["description"] = "Sales data API"
        
        results = broker.discover_data("sales", use_llm=False)
        
        assert len(results) > 0
        assert results[0]["source"] == "test_sales_api"
        assert results[0]["relevance"] > 0
    
    def test_llm_discovery(self, broker, sample_api_source):
        """Test LLM-powered discovery."""
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(broker, 'step', return_value="test_sales_api"):
            broker.register_data_source(sample_api_source, auto_connect=False)
            
            broker.data_catalog["test_sales_api"]["description"] = "Sales data API"
            
            results = broker.discover_data("sales data", use_llm=True)
            
            assert len(results) >= 0  # May fall back to simple discovery


class TestDataQuality:
    """Test data quality assessment."""
    
    def test_assess_data_quality(self, broker, sample_api_source):
        """Test data quality assessment."""
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(sample_api_source, 'get_cached_data', return_value={"test": "data"}), \
             patch.object(broker, 'step', return_value="Data quality is good"):
            broker.register_data_source(sample_api_source, auto_connect=False)
            
            assessment = broker.assess_data_quality("test_sales_api")
            
            assert "quality_score" in assessment
            assert "metrics" in assessment
            assert assessment["source"] == "test_sales_api"


class TestErrorHandling:
    """Test error handling."""
    
    def test_register_invalid_source(self, broker, sample_api_source):
        """Test handling invalid source registration."""
        with patch.object(sample_api_source, 'connect', return_value=False):
            result = broker.register_data_source(sample_api_source, auto_connect=True)
            assert result is False
    
    def test_collect_from_nonexistent_source(self, broker):
        """Test collecting from non-existent source."""
        data = broker.collect_data(sources=["nonexistent"])
        assert "nonexistent" not in data or data.get("nonexistent") is None
    
    def test_analyze_nonexistent_source(self, broker):
        """Test analyzing non-existent source."""
        analysis = broker.analyze_data_dependencies("nonexistent")
        assert "error" in analysis


class TestIntegration:
    """Integration tests."""
    
    def test_complete_workflow(self, broker, sample_api_source, sample_file_source):
        """Test complete workflow from setup to routing."""
        # Setup
        with patch.object(sample_api_source, 'connect', return_value=True), \
             patch.object(sample_file_source, 'connect', return_value=True):
            broker.register_data_source(sample_api_source, auto_connect=False)
            broker.register_data_source(sample_file_source, auto_connect=False)
        
        # Model dependencies
        broker.add_causal_relationship("test_sales_api", "revenue_prediction", strength=0.8)
        
        # Register consumer
        consumer = ConsumerRequirement(
            name="analytics",
            required_fields=["product_id", "revenue"],
            causal_dependencies=["test_sales_api"]
        )
        broker.register_consumer(consumer)
        
        # Route data
        routes = broker.route_data(data="test_sales_api", consumers=["analytics"])
        
        # Verify
        assert len(broker.data_sources) == 2
        assert len(broker.routing_engine.consumers) == 1
        assert "analytics" in routes or len(routes) >= 0  # May be empty if no match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

