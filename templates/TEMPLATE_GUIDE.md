# Template Framework Guide

## Overview

The Template Framework provides a standardized way to create specialized agents with LLM integration, graph management, prediction capabilities, and statistical analysis. It extracts common patterns from `CRCA.py` into reusable modules.

**✨ Drag-and-Drop Features**: Modules can be used as "drag-and-drop" features via mixins or the module registry, allowing you to quickly compose agents with only the features you need.

## Architecture

The framework consists of:

1. **BaseSpecializedAgent**: Base class with common initialization and memory patterns
2. **GraphManager**: Graph operations (nodes, edges, topological sort, path finding)
3. **PredictionFramework**: Prediction methods, standardization, counterfactual generation
4. **StatisticalMethods**: Data fitting, uncertainty quantification, time-series analysis
5. **LLMIntegration**: Schema definition, prompt building, multi-loop reasoning

## Quick Start

### Drag-and-Drop with Mixins (Recommended)

The easiest way to create agents is using feature mixins:

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import GraphFeatureMixin, PredictionFeatureMixin, LLMFeatureMixin

class MyAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize features - just "drag and drop" what you need!
        self.init_graph_feature(variables=["A", "B", "C"])
        self.init_prediction_feature(use_nonlinear=True)
        self.init_llm_feature()
    
    def _get_domain_schema(self):
        return {...}  # Your schema
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Task: {task}"
    
    def _domain_specific_setup(self):
        pass

# Use it
agent = MyAgent(agent_name="my-agent")
predictions = agent.predict({"A": 1.0}, {"A": 1.5})
analysis = agent.analyze_with_llm("Analyze this problem")
```

### Minimal Agent Example (LLM Only)

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import LLMFeatureMixin
from templates.llm_integration import create_default_schema

class MyAgent(BaseSpecializedAgent, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_llm_feature()  # Just LLM, nothing else
    
    def _get_domain_schema(self):
        return create_default_schema(
            function_name="my_analysis",
            description="My domain analysis",
            properties={
                "analysis": {"type": "string", "description": "Analysis result"}
            }
        )
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Analyze: {task}"
    
    def _domain_specific_setup(self):
        pass

# Use it
agent = MyAgent(agent_name="my-agent", model_name="gpt-4o")
result = agent.analyze_with_llm("Analyze this problem")
```

### Full-Featured Agent Example

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.graph_management import GraphManager
from templates.prediction_framework import PredictionFramework
from templates.statistical_methods import StatisticalMethods
from templates.llm_integration import LLMIntegration, create_default_schema

class FullAgent(BaseSpecializedAgent):
    def __init__(self, variables=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize modules
        self.graph_manager = GraphManager()
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager
        )
        self.statistical_methods = StatisticalMethods(
            graph_manager=self.graph_manager,
            prediction_framework=self.prediction_framework
        )
        self.llm_integration = LLMIntegration(agent=self)
        
        # Add variables and edges
        if variables:
            self.graph_manager.add_nodes_from(variables)
        if edges:
            self.graph_manager.add_edges_from(edges)
    
    def _get_domain_schema(self):
        return create_default_schema(...)
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Task: {task}"
    
    def _domain_specific_setup(self):
        pass
```

## Module Details

### BaseSpecializedAgent

**Purpose**: Provides common initialization, schema handling, and memory management.

**Key Methods**:
- `_get_domain_schema()`: Return tool schema (abstract, must implement)
- `_build_domain_prompt(task)`: Build domain-specific prompt (abstract, must implement)
- `_domain_specific_setup()`: Initialize domain-specific attributes (override as needed)
- `step(task)`: Execute single reasoning step
- `_build_memory_context()`: Build context from memory
- `clear_cache()`: Clear prediction cache

**Example**:
```python
class MyAgent(BaseSpecializedAgent):
    def _get_domain_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "my_function",
                "description": "...",
                "parameters": {...}
            }
        }
    
    def _build_domain_prompt(self, task: str) -> str:
        return f"Domain-specific prompt: {task}"
```

### GraphManager

**Purpose**: Manages graph operations (nodes, edges, paths, topological sort).

**Key Methods**:
- `ensure_node_exists(node)`: Add node if it doesn't exist
- `add_relationship(source, target, **metadata)`: Add edge with metadata
- `get_parents(node)`: Get parent nodes
- `get_children(node)`: Get child nodes
- `topological_sort()`: Get nodes in topological order
- `identify_path(start, end)`: Find path between nodes
- `is_dag()`: Check if graph is acyclic

**Example**:
```python
graph = GraphManager(graph_type="causal")
graph.add_nodes_from(["A", "B", "C"])
graph.add_relationship("A", "B", strength=0.8)
graph.add_relationship("B", "C", strength=0.6)
order = graph.topological_sort()  # ['A', 'B', 'C']
```

### PredictionFramework

**Purpose**: Provides prediction methods, standardization, and counterfactual generation.

**Key Methods**:
- `standardize_state(state)`: Standardize state values
- `predict_outcomes(factual_state, interventions)`: Predict outcomes
- `predict_outcomes_cached(...)`: Cached version
- `counterfactual_abduction_action_prediction(...)`: Counterfactual reasoning
- `generate_counterfactual_scenarios(...)`: Generate scenarios
- `set_standardization_stats(variable, mean, std)`: Set stats

**Example**:
```python
framework = PredictionFramework(graph_manager=graph)
outcomes = framework.predict_outcomes(
    factual_state={"A": 1.0, "B": 2.0},
    interventions={"A": 1.5}
)
scenarios = framework.generate_counterfactual_scenarios(
    factual_state={"A": 1.0},
    target_variables=["A"],
    max_scenarios=5
)
```

### StatisticalMethods

**Purpose**: Provides data-driven fitting and statistical analysis.

**Key Methods**:
- `fit_from_dataframe(df, variables, ...)`: Fit model from data
- `quantify_uncertainty(df, variables, ...)`: Bootstrap uncertainty
- `granger_causality_test(df, var1, var2)`: Granger causality
- `vector_autoregression_estimation(df, variables)`: VAR estimation
- `bayesian_edge_inference(df, parent, child)`: Bayesian inference
- `sensitivity_analysis(intervention, target)`: Sensitivity analysis
- `deep_root_cause_analysis(problem_variable)`: Root cause analysis
- `shapley_value_attribution(...)`: Shapley values

**Example**:
```python
stats = StatisticalMethods(graph_manager=graph, prediction_framework=framework)
stats.fit_from_dataframe(df, variables=["A", "B", "C"], window=30)
uncertainty = stats.quantify_uncertainty(df, variables=["A", "B"], windows=200)
```

### LLMIntegration

**Purpose**: Provides LLM integration patterns (schema, prompts, memory).

**Key Methods**:
- `build_domain_prompt(task)`: Build prompt (override in subclass)
- `run_llm_domain_analysis(task, ...)`: Run multi-loop analysis
- `build_memory_context()`: Build memory context
- `synthesize_analysis(task)`: Synthesize final analysis

**Example**:
```python
llm = LLMIntegration(agent=my_agent, max_loops=3)
result = llm.run_llm_domain_analysis(
    "Analyze this problem",
    build_prompt_fn=my_agent._build_domain_prompt
)
```

## Creating a New Specialized Agent

### Step 1: Define Your Agent Class

```python
from templates.base_specialized_agent import BaseSpecializedAgent

class MyDomainAgent(BaseSpecializedAgent):
    pass
```

### Step 2: Implement Required Methods

```python
class MyDomainAgent(BaseSpecializedAgent):
    def _get_domain_schema(self):
        """Return your tool schema."""
        return {
            "type": "function",
            "function": {
                "name": "my_domain_analysis",
                "description": "...",
                "parameters": {...}
            }
        }
    
    def _build_domain_prompt(self, task: str) -> str:
        """Build domain-specific prompt."""
        return f"My domain prompt: {task}"
    
    def _domain_specific_setup(self):
        """Initialize domain-specific attributes."""
        self.my_domain_data = {}
```

### Step 3: Add Modules as Needed

```python
class MyDomainAgent(BaseSpecializedAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add graph management if needed
        self.graph_manager = GraphManager()
        
        # Add prediction if needed
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager
        )
        
        # Add statistics if needed
        self.statistical_methods = StatisticalMethods(
            graph_manager=self.graph_manager,
            prediction_framework=self.prediction_framework
        )
```

### Step 4: Implement Domain Logic

```python
class MyDomainAgent(BaseSpecializedAgent):
    # ... previous code ...
    
    def my_domain_method(self, input_data):
        """Your domain-specific logic."""
        # Use modules
        predictions = self.prediction_framework.predict_outcomes(...)
        return predictions
    
    def run(self, task=None, **kwargs):
        """Override run() if needed."""
        if isinstance(task, str):
            return self.llm_integration.run_llm_domain_analysis(task)
        # ... handle other cases ...
```

## Best Practices

### 1. Composition Over Inheritance

Use composition with modules rather than deep inheritance:

```python
# Good: Composition
class MyAgent(BaseSpecializedAgent):
    def __init__(self):
        super().__init__()
        self.graph_manager = GraphManager()
        self.prediction_framework = PredictionFramework(...)

# Avoid: Deep inheritance
class MyAgent(SomeComplexBase):
    # Too many layers of inheritance
```

### 2. Keep Domain Logic Separate

Separate domain-specific logic from framework logic:

```python
class MyAgent(BaseSpecializedAgent):
    def _domain_specific_setup(self):
        # Domain-specific initialization
        self.domain_config = {...}
    
    def domain_specific_method(self):
        # Domain-specific logic
        pass
```

### 3. Use Type Hints

Always use type hints for better IDE support and documentation:

```python
def predict_outcomes(
    self,
    factual_state: Dict[str, float],
    interventions: Dict[str, float]
) -> Dict[str, float]:
    ...
```

### 4. Handle Optional Dependencies

Check for optional dependencies before using them:

```python
def my_method(self):
    try:
        import pandas as pd
        # Use pandas
    except ImportError:
        raise ImportError("pandas required for this method")
```

### 5. Document Your Schema

Provide clear descriptions in your schema:

```python
def _get_domain_schema(self):
    return create_default_schema(
        function_name="my_analysis",
        description="Clear description of what this does",
        properties={
            "result": {
                "type": "string",
                "description": "Clear description of this parameter"
            }
        }
    )
```

## Drag-and-Drop Feature Composition

### Available Features

The framework provides these features as mixins:

1. **GraphFeatureMixin**: Graph management (nodes, edges, paths)
2. **PredictionFeatureMixin**: Prediction and counterfactual generation
3. **StatisticsFeatureMixin**: Statistical analysis and data fitting
4. **LLMFeatureMixin**: LLM integration and multi-loop reasoning
5. **FullFeatureMixin**: All features combined

### Mixin Pattern (Recommended)

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import GraphFeatureMixin, PredictionFeatureMixin

class MyAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize only the features you need
        self.init_graph_feature(variables=["A", "B"])
        self.init_prediction_feature()
    
    # ... implement required methods ...
```

### Programmatic Composition

```python
from templates.module_registry import compose_agent

# Create agent class with selected features
AgentClass = compose_agent(
    BaseSpecializedAgent,
    features=['graph', 'prediction'],
    feature_configs={
        'graph': {'graph_type': 'causal'},
        'prediction': {'use_nonlinear': True}
    }
)

# Use it
agent = AgentClass(agent_name="my-agent")
```

### Feature Combinations

**Minimal (LLM only)**:
```python
class MinimalAgent(BaseSpecializedAgent, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_llm_feature()
```

**Graph + LLM**:
```python
class GraphLLMAgent(BaseSpecializedAgent, GraphFeatureMixin, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_graph_feature()
        self.init_llm_feature()
```

**Graph + Prediction (No LLM)**:
```python
class PredictionAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_graph_feature()
        self.init_prediction_feature()
```

**All Features**:
```python
class FullAgent(BaseSpecializedAgent, FullFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_all_features()  # One call for everything
```

## Examples

See the `examples/` directory for complete examples:

- `causal_agent_template.py`: Full causal reasoning agent
- `trading_agent_template.py`: Trading/financial agent
- `logistics_agent_template.py`: Logistics/supply chain agent
- `drag_drop_example.py`: **Drag-and-drop composition examples**

## Migration from CRCA.py

If you're migrating from the original `CRCA.py`:

1. **Identify your domain**: What type of agent are you creating?
2. **Map to modules**: Which modules do you need?
   - Graph operations → `GraphManager`
   - Predictions → `PredictionFramework`
   - Statistics → `StatisticalMethods`
   - LLM integration → `LLMIntegration`
3. **Extract domain logic**: Keep only domain-specific code
4. **Test compatibility**: Ensure API compatibility if needed

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install rustworkx pandas scipy numpy
```

### Module Not Found

If modules aren't found, check your Python path:

```python
import sys
sys.path.insert(0, '/path/to/templates')
```

### Graph Operations Failing

If graph operations fail, check if rustworkx is installed. The framework falls back to dict-based operations if rustworkx is unavailable.

### Prediction Errors

If predictions fail, ensure standardization stats are set:

```python
framework.set_standardization_stats("variable", mean=0.0, std=1.0)
```

## Advanced Usage

### Custom Graph Types

You can create custom graph types by subclassing `GraphManager`:

```python
class MyGraphManager(GraphManager):
    def __init__(self):
        super().__init__(graph_type="my_type")
        # Add custom graph logic
```

### Custom Prediction Models

Override prediction methods in `PredictionFramework`:

```python
class MyPredictionFramework(PredictionFramework):
    def predict_outcomes(self, state, interventions):
        # Custom prediction logic
        return super().predict_outcomes(state, interventions)
```

### Async Operations

The framework supports async operations:

```python
import asyncio

async def run_async():
    result = await agent.run_async(task="...")
```

## Contributing

When adding new features:

1. Keep modules focused and single-purpose
2. Maintain backward compatibility
3. Add type hints and docstrings
4. Update this guide
5. Add examples

## License

Same as the main project.

