# Drag-and-Drop Feature Composition Guide

## Yes! Modules Can Be Used as Drag-and-Drop Features

The template framework now supports **true drag-and-drop composition** of features. You can pick and choose exactly which capabilities your agent needs.

## Two Ways to Compose Features

### Method 1: Mixin Pattern (Easiest)

Simply inherit from the mixins you want:

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import (
    GraphFeatureMixin,
    PredictionFeatureMixin,
    LLMFeatureMixin
)

class MyAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize only what you need
        self.init_graph_feature(variables=["A", "B"])
        self.init_prediction_feature()
        self.init_llm_feature()
    
    # ... implement required methods ...
```

**Available Mixins:**
- `GraphFeatureMixin` - Graph management
- `PredictionFeatureMixin` - Predictions and counterfactuals
- `StatisticsFeatureMixin` - Statistical analysis
- `LLMFeatureMixin` - LLM integration
- `FullFeatureMixin` - All features at once

### Method 2: Programmatic Composition

Use the module registry to compose features programmatically:

```python
from templates.module_registry import compose_agent

# Create agent class with selected features
AgentClass = compose_agent(
    BaseSpecializedAgent,
    features=['graph', 'prediction', 'llm'],
    feature_configs={
        'graph': {'graph_type': 'causal'},
        'prediction': {'use_nonlinear': True}
    }
)

agent = AgentClass(agent_name="my-agent")
```

## Feature Dependencies

Some features have dependencies (automatically handled):

- `PredictionFeatureMixin` requires `GraphFeatureMixin`
- `StatisticsFeatureMixin` requires both `GraphFeatureMixin` and `PredictionFeatureMixin`
- `LLMFeatureMixin` is standalone (no dependencies)

## Examples

### Minimal Agent (LLM Only)

```python
class MinimalAgent(BaseSpecializedAgent, LLMFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_llm_feature()  # Just LLM, nothing else
```

### Graph + Prediction (No LLM, No Statistics)

```python
class PredictionAgent(BaseSpecializedAgent, GraphFeatureMixin, PredictionFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_graph_feature()
        self.init_prediction_feature()
    
    def run(self, state, interventions):
        return self.predict(state, interventions)  # Direct predictions
```

### All Features

```python
class FullAgent(BaseSpecializedAgent, FullFeatureMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_all_features()  # One call for everything
```

## Benefits

1. **Flexibility**: Pick only what you need
2. **Lightweight**: No unnecessary dependencies
3. **Composable**: Mix and match features
4. **Type-Safe**: Full type hints and IDE support
5. **Easy**: Simple mixin inheritance pattern

## See Also

- `templates/examples/drag_drop_example.py` - Complete examples
- `templates/TEMPLATE_GUIDE.md` - Full framework documentation

