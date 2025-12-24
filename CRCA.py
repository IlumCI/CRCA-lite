"""CRCAAgent - Causal Reasoning and Counterfactual Analysis Agent.

This module provides a lightweight causal reasoning agent with LLM integration,
implemented in pure Python and intended as a flexible CR-CA engine for Swarms.
"""

# Standard library imports
import asyncio
import importlib
import inspect
import logging
import math
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
from loguru import logger
from swarms.structs.agent import Agent

# Try to import rustworkx (required dependency)
try:
    import rustworkx as rx
except Exception as e:
    raise ImportError(
        "rustworkx is required for the CRCAAgent rustworkx upgrade: pip install rustworkx"
    ) from e

# Optional heavy dependencies — used when available
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    from scipy import linalg as scipy_linalg  # type: ignore
    from scipy import stats as scipy_stats  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import cvxpy as cp  # type: ignore
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    import os

    # Explicitly load from .env file in the project root
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path, override=False)
except ImportError:
    # dotenv not available, skip loading
    pass
except Exception:
    # .env file might not exist, that's okay
    pass

# Local imports
try:
    from prompts.default_crca import DEFAULT_CRCA_SYSTEM_PROMPT
except ImportError:
    # Fallback if prompt file doesn't exist
    DEFAULT_CRCA_SYSTEM_PROMPT = None

# Fix litellm async compatibility
try:
    lu_spec = importlib.util.find_spec("litellm.litellm_core_utils.logging_utils")
    if lu_spec is not None:
        lu = importlib.import_module("litellm.litellm_core_utils.logging_utils")
        try:
            if hasattr(lu, "asyncio") and hasattr(lu.asyncio, "iscoroutinefunction"):
                lu.asyncio.iscoroutinefunction = inspect.iscoroutinefunction
        except Exception:
            pass
except Exception:
    pass


class CausalRelationType(Enum):
    """Enumeration of causal relationship types.
    
    Defines the different types of causal relationships that can exist
    between variables in a causal graph.
    """
    
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class CausalNode:
    """Represents a node in the causal graph.
    
    Attributes:
        name: Name of the variable/node
        value: Current value of the variable (optional)
        confidence: Confidence level in the node (default: 1.0)
        node_type: Type of the node (default: "variable")
    """
    
    name: str
    value: Optional[float] = None
    confidence: float = 1.0
    node_type: str = "variable"


@dataclass
class CausalEdge:
    """Represents an edge (causal relationship) in the causal graph.
    
    Attributes:
        source: Source variable name
        target: Target variable name
        strength: Strength of the causal relationship (default: 1.0)
        relation_type: Type of causal relation (default: DIRECT)
        confidence: Confidence level in the edge (default: 1.0)
    """
    
    source: str
    target: str
    strength: float = 1.0
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    confidence: float = 1.0


@dataclass
class CounterfactualScenario:
    """Represents a counterfactual scenario for analysis.
    
    Attributes:
        name: Name/identifier of the scenario
        interventions: Dictionary mapping variable names to intervention values
        expected_outcomes: Dictionary mapping variable names to expected outcomes
        probability: Probability of this scenario (default: 1.0)
        reasoning: Explanation of why this scenario is important
    """
    
    name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float = 1.0
    reasoning: str = ""


class CRCAAgent(Agent):
    """Causal Reasoning with Counterfactual Analysis Agent.
    
    A lightweight causal reasoning agent with LLM integration, providing both
    LLM-based causal analysis and deterministic causal simulation. Supports
    automatic variable extraction, causal graph management, counterfactual
    scenario generation, and comprehensive causal analysis.
    
    Key Features:
        - LLM integration for sophisticated causal reasoning
        - Dual-mode operation: LLM-based analysis and deterministic simulation
        - Automatic variable extraction from natural language tasks
        - Causal graph management with rustworkx backend
        - Counterfactual scenario generation
        - Batch prediction support
        - Async/await support for concurrent operations
    
    Attributes:
        causal_graph: Dictionary representing the causal graph structure
        causal_memory: List storing analysis steps and results
        causal_max_loops: Maximum number of loops for causal reasoning
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 3,
        agent_name: str = "cr-ca-lite-agent",
        agent_description: str = "Lightweight Causal Reasoning with Counterfactual Analysis Agent",
        description: Optional[str] = None,
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        enable_batch_predict: bool = False,
        max_batch_size: int = 32,
        bootstrap_workers: int = 0,
        use_async: bool = False,
        seed: Optional[int] = None,
        enable_excel: bool = False,
        agent_max_loops: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        """
        Initialize CRCAAgent with causal reasoning capabilities.
        
        Args:
            variables: List of variable names for the causal graph
            causal_edges: List of (source, target) tuples defining causal relationships
            max_loops: Maximum loops for causal reasoning (default: 3)
            agent_max_loops: Maximum loops for standard Agent operations (supports "auto")
                            If not provided, defaults to 1 for individual LLM calls.
                            Pass "auto" to enable automatic loop detection for standard operations.
            **kwargs: Additional arguments passed to parent Agent class
        """
        
        cr_ca_schema = CRCAAgent._get_cr_ca_schema()
        extract_variables_schema = CRCAAgent._get_extract_variables_schema()
        
        # Backwards-compatible alias for description
        agent_description = description or agent_description

        # Handle max_loops for standard Agent operations
        # agent_max_loops parameter takes precedence, then check kwargs, then default to 1
        if agent_max_loops is None:
            agent_max_loops = kwargs.pop("max_loops", 1)
        else:
            # Remove max_loops from kwargs if agent_max_loops was explicitly provided
            kwargs.pop("max_loops", None)

        # Merge tools_list_dictionary from kwargs with CRCA schema
        # Only add CRCA schema if user hasn't explicitly disabled it
        use_crca_tools = kwargs.pop("use_crca_tools", True)  # Default to True for backwards compatibility
        existing_tools = kwargs.pop("tools_list_dictionary", [])
        if not isinstance(existing_tools, list):
            existing_tools = [existing_tools] if existing_tools else []
        
        # Only add CRCA schemas if enabled
        if use_crca_tools:
            tools_list = [cr_ca_schema, extract_variables_schema] + existing_tools
        else:
            tools_list = existing_tools
        
        # Get existing callable tools (functions) from kwargs
        existing_callable_tools = kwargs.pop("tools", [])
        if not isinstance(existing_callable_tools, list):
            existing_callable_tools = [existing_callable_tools] if existing_callable_tools else []

        agent_kwargs = {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "model_name": model_name,
            "max_loops": agent_max_loops,  # Use user-provided value or default to 1
            "output_type": "final",
            **kwargs,  # All other Agent parameters passed through
        }
        
        # Always provide tools_list_dictionary if we have schemas
        # This ensures the LLM knows about the tools
        if tools_list:
            agent_kwargs["tools_list_dictionary"] = tools_list
        
        # Add existing callable tools if any
        if existing_callable_tools:
            agent_kwargs["tools"] = existing_callable_tools
        
        # Always apply default CRCA prompt as base, then add custom prompt on top if provided
        final_system_prompt = None
        if DEFAULT_CRCA_SYSTEM_PROMPT is not None:
            if system_prompt is not None:
                # Combine: default first, then custom on top
                final_system_prompt = f"{DEFAULT_CRCA_SYSTEM_PROMPT}\n\n--- Additional Instructions ---\n{system_prompt}"
            else:
                # Just use default
                final_system_prompt = DEFAULT_CRCA_SYSTEM_PROMPT
        elif system_prompt is not None:
            # If no default but custom prompt provided, use custom prompt
            final_system_prompt = system_prompt
        
        if final_system_prompt is not None:
            agent_kwargs["system_prompt"] = final_system_prompt
        if global_system_prompt is not None:
            agent_kwargs["global_system_prompt"] = global_system_prompt
        if secondary_system_prompt is not None:
            agent_kwargs["secondary_system_prompt"] = secondary_system_prompt
        
        super().__init__(**agent_kwargs)
        
        # Now that self exists, create and add the CRCA tool handlers if tools are enabled
        if use_crca_tools:
            # Create a wrapper function with the correct name that matches the schema
            def generate_causal_analysis(
                causal_analysis: str,
                intervention_planning: str,
                counterfactual_scenarios: List[Dict[str, Any]],
                causal_strength_assessment: str,
                optimal_solution: str,
            ) -> Dict[str, Any]:
                """Tool handler for generate_causal_analysis - wrapper that calls the instance method."""
                return self._generate_causal_analysis_handler(
                    causal_analysis=causal_analysis,
                    intervention_planning=intervention_planning,
                    counterfactual_scenarios=counterfactual_scenarios,
                    causal_strength_assessment=causal_strength_assessment,
                    optimal_solution=optimal_solution,
                )
            
            # Create a wrapper function for extract_causal_variables
            def extract_causal_variables(
                required_variables: List[str],
                causal_edges: List[List[str]],
                reasoning: str,
                optional_variables: Optional[List[str]] = None,
                counterfactual_variables: Optional[List[str]] = None,
            ) -> Dict[str, Any]:
                """Tool handler for extract_causal_variables - wrapper that calls the instance method."""
                return self._extract_causal_variables_handler(
                    required_variables=required_variables,
                    causal_edges=causal_edges,
                    reasoning=reasoning,
                    optional_variables=optional_variables,
                    counterfactual_variables=counterfactual_variables,
                )
            
            # Add the wrapper functions to the tools list
            # The function names must match the schema names
            if self.tools is None:
                self.tools = []
            self.add_tool(generate_causal_analysis)
            self.add_tool(extract_causal_variables)
            
            # CRITICAL: Re-initialize tool_struct after adding tools
            # This ensures the BaseTool instance has the updated tools and function_map
            if hasattr(self, 'setup_tools'):
                self.tool_struct = self.setup_tools()
            
            # Ensure tools_list_dictionary is set with our manual schemas
            if not self.tools_list_dictionary or len(self.tools_list_dictionary) == 0:
                self.tools_list_dictionary = [cr_ca_schema, extract_variables_schema] + existing_tools
            else:
                # Replace any auto-generated schemas with our manual ones
                our_tool_names = {"generate_causal_analysis", "extract_causal_variables"}
                filtered = []
                for schema in self.tools_list_dictionary:
                    if isinstance(schema, dict):
                        func_name = schema.get("function", {}).get("name", "")
                        if func_name in our_tool_names:
                            continue  # Skip auto-generated, we'll add manual
                    filtered.append(schema)
                # Add our manual schemas first, then existing tools
                self.tools_list_dictionary = [cr_ca_schema, extract_variables_schema] + filtered
        
        self.causal_max_loops = max_loops
        self.causal_graph: Dict[str, Dict[str, float]] = {}
        self.causal_graph_reverse: Dict[str, List[str]] = {}  # For fast parent lookup
        self._graph = rx.PyDiGraph()
        self._node_to_index: Dict[str, int] = {}
        self._index_to_node: Dict[int, str] = {}
        
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        self.use_nonlinear_scm: bool = True
        self.nonlinear_activation: str = "tanh"  # options: 'tanh'|'identity'
        self.interaction_terms: Dict[str, List[Tuple[str, str]]] = {}
        self.edge_sign_constraints: Dict[Tuple[str, str], int] = {}
        self.bayesian_priors: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.enable_batch_predict = bool(enable_batch_predict)
        self.max_batch_size = int(max_batch_size)
        self.bootstrap_workers = int(max(0, bootstrap_workers))
        self.use_async = bool(use_async)
        self.seed = seed if seed is not None else 42
        self._rng = np.random.default_rng(self.seed)
        
        if variables:
            for var in variables:
                self._ensure_node_exists(var)

        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

        self.causal_memory: List[Dict[str, Any]] = []
        self._prediction_cache: Dict[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]], Dict[str, float]] = {}
        self._prediction_cache_order: List[Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = []
        self._prediction_cache_max: int = 1000
        self._cache_enabled: bool = True
        self._prediction_cache_lock = threading.Lock()
        
        # Excel TUI integration
        self._excel_enabled = bool(enable_excel)
        self._excel_tables = None
        self._excel_eval_engine = None
        self._excel_scm_bridge = None
        self._excel_dependency_graph = None
        
        if enable_excel:
            try:
                from crca_excel.core.tables import TableManager
                from crca_excel.core.deps import DependencyGraph
                from crca_excel.core.eval import EvaluationEngine
                from crca_excel.core.scm import SCMBridge
                
                self._excel_tables = TableManager()
                self._excel_dependency_graph = DependencyGraph()
                self._excel_eval_engine = EvaluationEngine(
                    self._excel_tables,
                    self._excel_dependency_graph,
                    max_iter=self.causal_max_loops if isinstance(self.causal_max_loops, int) else 100,
                    epsilon=1e-6
                )
                self._excel_scm_bridge = SCMBridge(
                    self._excel_tables,
                    self._excel_eval_engine,
                    crca_agent=self
                )
                
                # Initialize standard tables
                self._initialize_excel_tables()
                
                # Link causal graph to tables
                self._link_causal_graph_to_tables()
            except ImportError as e:
                logger.warning(f"Excel TUI modules not available: {e}")
                self._excel_enabled = False

    @staticmethod
    def _get_cr_ca_schema() -> Dict[str, Any]:
        
        return {
            "type": "function",
            "function": {
                "name": "generate_causal_analysis",
                "description": "Generates structured causal reasoning and counterfactual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "causal_analysis": {
                            "type": "string",
                            "description": "Analysis of causal relationships and mechanisms"
                        },
                        "intervention_planning": {
                            "type": "string", 
                            "description": "Planned interventions to test causal hypotheses"
                        },
                        "counterfactual_scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scenario_name": {"type": "string"},
                                    "interventions": {"type": "object"},
                                    "expected_outcomes": {"type": "object"},
                                    "reasoning": {"type": "string"}
                                }
                            },
                            "description": "Multiple counterfactual scenarios to explore"
                        },
                        "causal_strength_assessment": {
                            "type": "string",
                            "description": "Assessment of causal relationship strengths and confounders"
                        },
                        "optimal_solution": {
                            "type": "string",
                            "description": "Recommended optimal solution based on causal analysis"
                        }
                    },
                    "required": [
                        "causal_analysis",
                        "intervention_planning", 
                        "counterfactual_scenarios",
                        "causal_strength_assessment",
                        "optimal_solution"
                    ]
                }
            }
        }

    @staticmethod
    def _get_extract_variables_schema() -> Dict[str, Any]:
        """
        Get the schema for the extract_causal_variables tool.
        
        Returns:
            Dictionary containing the OpenAI function schema for variable extraction
        """
        return {
            "type": "function",
            "function": {
                "name": "extract_causal_variables",
                "description": "Extract and propose causal variables, relationships, and counterfactual scenarios needed for causal analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "required_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Core variables that must be included for causal analysis"
                        },
                        "optional_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional variables that may be useful but not essential"
                        },
                        "causal_edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Causal relationships as [source, target] pairs"
                        },
                        "counterfactual_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Variables to explore in counterfactual scenarios"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why these variables and relationships are needed"
                        }
                    },
                    "required": ["required_variables", "causal_edges", "reasoning"]
                }
            }
        }

    def step(self, task: str) -> str:
        """
        Execute a single step of causal reasoning.
        
        Args:
            task: Task string to process
            
        Returns:
            Response string from the agent
        """
        response = super().run(task)
        return response
    
    def _generate_causal_analysis_handler(
        self,
        causal_analysis: str,
        intervention_planning: str,
        counterfactual_scenarios: List[Dict[str, Any]],
        causal_strength_assessment: str,
        optimal_solution: str,
    ) -> Dict[str, Any]:
        """
        Handler function for the generate_causal_analysis tool.
        
        This function is called when the LLM invokes the generate_causal_analysis tool.
        It processes the tool's output and integrates it into the causal graph.
        
        Args:
            causal_analysis: Analysis of causal relationships and mechanisms
            intervention_planning: Planned interventions to test causal hypotheses
            counterfactual_scenarios: List of counterfactual scenarios
            causal_strength_assessment: Assessment of causal relationship strengths
            optimal_solution: Recommended optimal solution
            
        Returns:
            Dictionary with processed results
        """
        logger.info("Processing causal analysis from tool call")
        
        # Store the analysis in causal memory
        analysis_entry = {
            'type': 'analysis',  # Mark this as an analysis entry
            'causal_analysis': causal_analysis,
            'intervention_planning': intervention_planning,
            'counterfactual_scenarios': counterfactual_scenarios,
            'causal_strength_assessment': causal_strength_assessment,
            'optimal_solution': optimal_solution,
            'timestamp': len(self.causal_memory)
        }
        
        self.causal_memory.append(analysis_entry)
        
        # Try to extract and update causal relationships from the analysis
        # This is a simple implementation - can be enhanced later
        try:
            # The LLM might mention relationships in the analysis
            # For now, we'll just store the analysis
            # In a more advanced version, we could parse the analysis to extract relationships
            pass
        except Exception as e:
            logger.warning(f"Error processing causal analysis: {e}")
        
        # Return a structured response
        return {
            "status": "success",
            "message": "Causal analysis processed and stored",
            "analysis_summary": {
                "causal_analysis_length": len(causal_analysis),
                "num_scenarios": len(counterfactual_scenarios),
                "has_optimal_solution": bool(optimal_solution)
            }
        }

    def _extract_causal_variables_handler(
        self,
        required_variables: List[str],
        causal_edges: List[List[str]],
        reasoning: str,
        optional_variables: Optional[List[str]] = None,
        counterfactual_variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Handler function for the extract_causal_variables tool.
        
        This function is called when the LLM invokes the extract_causal_variables tool.
        It processes the tool's output and adds variables and edges to the causal graph.
        
        Args:
            required_variables: Core variables that must be included for causal analysis
            causal_edges: Causal relationships as [source, target] pairs
            reasoning: Explanation of why these variables and relationships are needed
            optional_variables: Additional variables that may be useful but not essential
            counterfactual_variables: Variables to explore in counterfactual scenarios
            
        Returns:
            Dictionary with processed results and summary of what was added
        """
        import traceback
        
        try:
            logger.info("=== EXTRACT HANDLER CALLED ===")
            logger.info(f"Processing variable extraction from tool call")
            logger.info(f"Received required_variables: {required_variables} (type: {type(required_variables)})")
            logger.info(f"Received causal_edges: {causal_edges} (type: {type(causal_edges)})")
            logger.info(f"Received optional_variables: {optional_variables}")
            logger.info(f"Received counterfactual_variables: {counterfactual_variables}")
            
            # Validate inputs
            if not required_variables:
                logger.warning("No required_variables provided!")
                required_variables = []
            if not isinstance(required_variables, list):
                logger.warning(f"required_variables is not a list: {type(required_variables)}")
                required_variables = [str(required_variables)] if required_variables else []
            
            if not causal_edges:
                logger.warning("No causal_edges provided!")
                causal_edges = []
            if not isinstance(causal_edges, list):
                logger.warning(f"causal_edges is not a list: {type(causal_edges)}")
                causal_edges = []
            
            # Track what was added
            added_variables = []
            added_edges = []
            skipped_edges = []
            
            # Add required variables
            for var in required_variables:
                if var and var.strip():
                    var_clean = var.strip()
                    if var_clean not in self.causal_graph:
                        self._ensure_node_exists(var_clean)
                        added_variables.append(var_clean)
            
            # Add optional variables
            if optional_variables:
                for var in optional_variables:
                    if var and var.strip():
                        var_clean = var.strip()
                        if var_clean not in self.causal_graph:
                            self._ensure_node_exists(var_clean)
                            added_variables.append(var_clean)
            
            # Add counterfactual variables (also add them as nodes)
            if counterfactual_variables:
                for var in counterfactual_variables:
                    if var and var.strip():
                        var_clean = var.strip()
                        if var_clean not in self.causal_graph:
                            self._ensure_node_exists(var_clean)
                            if var_clean not in added_variables:
                                added_variables.append(var_clean)
            
            # Add causal edges
            for edge in causal_edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    source = str(edge[0]).strip() if edge[0] else None
                    target = str(edge[1]).strip() if edge[1] else None
                    
                    if source and target:
                        # Ensure both nodes exist
                        if source not in self.causal_graph:
                            self._ensure_node_exists(source)
                            if source not in added_variables:
                                added_variables.append(source)
                        if target not in self.causal_graph:
                            self._ensure_node_exists(target)
                            if target not in added_variables:
                                added_variables.append(target)
                        
                        # Add the edge
                        try:
                            self.add_causal_relationship(source, target)
                            added_edges.append((source, target))
                        except Exception as e:
                            logger.warning(f"Failed to add edge {source} -> {target}: {e}")
                            skipped_edges.append((source, target))
                else:
                    logger.warning(f"Invalid edge format: {edge}")
                    skipped_edges.append(edge)
            
            # Store extraction metadata in causal memory
            extraction_entry = {
                'type': 'variable_extraction',
                'required_variables': required_variables,
                'optional_variables': optional_variables or [],
                'counterfactual_variables': counterfactual_variables or [],
                'causal_edges': causal_edges,
                'reasoning': reasoning,
                'added_variables': added_variables,
                'added_edges': added_edges,
                'skipped_edges': skipped_edges,
                'timestamp': len(self.causal_memory)
            }
            
            self.causal_memory.append(extraction_entry)
            
            # Log what was actually added
            logger.info(f"Extraction complete: Added {len(added_variables)} variables, {len(added_edges)} edges")
            logger.info(f"Added variables: {added_variables}")
            logger.info(f"Added edges: {added_edges}")
            logger.info(f"Current graph size: {len(self.causal_graph)} variables, {sum(len(children) for children in self.causal_graph.values())} edges")
            
            # Return a structured response
            result = {
                "status": "success",
                "message": "Variables and relationships extracted and added to causal graph",
                "summary": {
                    "variables_added": len(added_variables),
                    "edges_added": len(added_edges),
                    "edges_skipped": len(skipped_edges),
                    "total_variables_in_graph": len(self.causal_graph),
                    "total_edges_in_graph": sum(len(children) for children in self.causal_graph.values())
                },
                "details": {
                    "added_variables": added_variables,
                    "added_edges": added_edges,
                    "skipped_edges": skipped_edges if skipped_edges else None
                }
            }
            logger.info(f"Returning result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"ERROR in _extract_causal_variables_handler: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return error but don't fail completely
            return {
                "status": "error",
                "message": f"Error processing variable extraction: {str(e)}",
                "summary": {
                    "variables_added": 0,
                    "edges_added": 0,
                }
            }

    def _build_causal_prompt(self, task: str) -> str:
        return (
            "You are a Causal Reasoning with Counterfactual Analysis (CR-CA) agent.\n"
            f"Problem: {task}\n"
            f"Current causal graph has {len(self.causal_graph)} variables and "
            f"{sum(len(children) for children in self.causal_graph.values())} relationships.\n\n"
            "CRITICAL: You MUST use the generate_causal_analysis tool to provide your analysis.\n"
            "The variables have already been extracted. Now you need to generate the causal analysis.\n"
            "Do NOT call extract_causal_variables again. You MUST call generate_causal_analysis with:\n"
            "- causal_analysis: Detailed analysis of causal relationships and mechanisms\n"
            "- intervention_planning: Planned interventions to test causal hypotheses\n"
            "- counterfactual_scenarios: List of what-if scenarios (array of objects)\n"
            "- causal_strength_assessment: Assessment of relationship strengths and confounders\n"
            "- optimal_solution: Recommended solution based on analysis\n"
        )

    def _build_variable_extraction_prompt(self, task: str) -> str:
        """
        Build a prompt that guides the LLM to extract variables from a task.
        
        Args:
            task: The task string to analyze
            
        Returns:
            Formatted prompt string for variable extraction
        """
        return (
            "You are analyzing a causal reasoning task. The causal graph is currently empty.\n"
            f"Task: {task}\n\n"
            "**CRITICAL: You MUST use the extract_causal_variables tool to proceed.**\n"
            "Do NOT just describe what variables might be needed - you MUST call the tool.\n\n"
            "Call the extract_causal_variables tool with:\n"
            "1. required_variables: List of core variables needed for causal analysis\n"
            "2. causal_edges: List of [source, target] pairs showing causal relationships\n"
            "3. reasoning: Explanation of why these variables are needed\n"
            "4. optional_variables: (optional) Additional variables that may be useful\n"
            "5. counterfactual_variables: (optional) Variables to explore in what-if scenarios\n\n"
            "Example: For a pricing task, you might extract:\n"
            "- required_variables: ['price', 'demand', 'supply', 'cost']\n"
            "- causal_edges: [['price', 'demand'], ['cost', 'price'], ['supply', 'price']]\n"
            "- reasoning: 'Price affects demand, cost affects price, supply affects price'\n\n"
            "You can call the tool multiple times to refine your extraction.\n"
            "Be thorough - extract all variables and relationships implied by the task."
        )

    def _build_memory_context(self) -> str:
        """Build memory context from causal_memory, handling different memory entry structures."""
        context_parts = []
        for step in self.causal_memory[-2:]:  # Last 2 steps
            if isinstance(step, dict):
                # Handle standard analysis step structure
                if 'step' in step and 'analysis' in step:
                    context_parts.append(f"Step {step['step']}: {step['analysis']}")
                # Handle extraction entry structure
                elif 'type' in step and step.get('type') == 'extraction':
                    context_parts.append(f"Variable Extraction: {step.get('summary', 'Variables extracted')}")
                # Handle generic entry
                elif 'analysis' in step:
                    context_parts.append(f"Analysis: {step['analysis']}")
                elif 'summary' in step:
                    context_parts.append(f"Summary: {step['summary']}")
        return "\n".join(context_parts) if context_parts else ""

    def _synthesize_causal_analysis(self, task: str) -> str:
        """
        Synthesize a final causal analysis report from the analysis steps.
        
        Uses direct LLM call to avoid Agent tool execution issues.
        
        Args:
            task: Original task string
            
        Returns:
            Synthesized causal analysis report
        """
        synthesis_prompt = f"Based on the causal analysis steps performed, synthesize a concise causal report for: {task}"
        try:
            # Use direct LLM call to avoid tool execution errors
            response = self._call_llm_directly(synthesis_prompt)
            return str(response) if response else "Analysis synthesis failed"
        except Exception as e:
            logger.error(f"Error synthesizing causal analysis: {e}")
            return "Analysis synthesis failed"
    
    def _should_trigger_causal_analysis(self, task: Optional[Union[str, Any]]) -> bool:
        """
        Automatically detect if a task should trigger causal analysis.
        
        This method analyzes the task content to determine if it requires
        causal reasoning, counterfactual analysis, or relationship analysis.
        
        Args:
            task: The task string to analyze
            
        Returns:
            True if causal analysis should be triggered, False otherwise
        """
        if task is None:
            return False
        
        if not isinstance(task, str):
            return False
        
        task_lower = task.lower().strip()
        
        # Keywords that indicate causal analysis is needed
        causal_keywords = [
            # Causal relationship terms
            'causal', 'causality', 'cause', 'causes', 'caused by', 'causing',
            'relationship', 'relationships', 'relate', 'relates', 'related',
            'influence', 'influences', 'influenced', 'affect', 'affects', 'affected',
            'impact', 'impacts', 'impacted', 'effect', 'effects',
            'depend', 'depends', 'dependency', 'dependencies',
            'correlation', 'correlate', 'correlates',
            
            # Counterfactual analysis terms
            'counterfactual', 'counterfactuals', 'what if', 'what-if',
            'scenario', 'scenarios', 'alternative', 'alternatives',
            'hypothetical', 'hypothesis', 'hypotheses',
            'if then', 'if-then', 'suppose', 'assuming',
            
            # Prediction and forecasting terms (often need causal reasoning)
            'predict', 'prediction', 'forecast', 'forecasting', 'project', 'projection',
            'expected', 'expect', 'expectation', 'estimate', 'estimation',
            'future', 'future value', 'future price', 'in 24 months', 'in X months',
            'will be', 'would be', 'could be', 'might be',
            
            # Analysis terms
            'analyze', 'analysis', 'analyzing', 'analyze the', 'analyze how',
            'understand', 'understanding', 'explain', 'explanation',
            'reasoning', 'reason', 'rationale',
            
            # Relationship-specific terms
            'between', 'among', 'link', 'links', 'connection', 'connections',
            'chain', 'chains', 'path', 'paths', 'flow', 'flows',
            
            # Intervention terms
            'intervention', 'interventions', 'change', 'changes', 'modify', 'modifies',
            'adjust', 'adjusts', 'alter', 'alters',
            
            # Risk and consequence terms (NEW)
            'risk', 'risks', 'consequence', 'consequences', 'benefit', 'benefits',
            'trade-off', 'trade-offs', 'tradeoff', 'tradeoffs',
            'downside', 'downsides', 'upside', 'upsides',
            
            # Determination and outcome terms (NEW)
            'determine', 'determines', 'determining', 'determination',
            'result', 'results', 'resulting', 'outcome', 'outcomes',
            'consideration', 'considerations', 'factor', 'factors',
            'driver', 'drivers', 'driving', 'drives', 'driven',
            'lead to', 'leads to', 'leading to', 'led to',
            
            # Decision-making terms (NEW)
            'should', 'should we', 'should i', 'should they',
            'better', 'best', 'worse', 'worst', 'compare', 'comparison',
            'option', 'options', 'choice', 'choices', 'choose',
            'strategy', 'strategies', 'approach', 'approaches',
            'decision', 'decisions', 'decide',
            
            # Importance and consideration terms (NEW)
            'important', 'importance', 'matter', 'matters', 'mattering',
            'consider', 'considering', 'consideration', 'considerations',
            'key', 'keys', 'critical', 'crucial', 'essential',
        ]
        
        # Check if task contains causal keywords
        for keyword in causal_keywords:
            if keyword in task_lower:
                return True
        
        # Check for questions about variables in the causal graph
        # If the task mentions any of our variables, it's likely a causal question
        graph_variables = [var.lower() for var in self.causal_graph.keys()]
        for var in graph_variables:
            if var in task_lower:
                return True
        
        # Check for patterns that suggest causal reasoning
        causal_patterns = [
            'how does', 'how do', 'how will', 'how would', 'how might', 'how can',
            'why does', 'why do', 'why will', 'why would', 'why did',
            'what happens if', 'what would happen', 'what will happen', 'what happens when',
            'if we', 'if you', 'if they', 'if it', 'if this', 'if that',
            'what should', 'what should we', 'what should i',
            'which is', 'which are', 'which would', 'which will',
            'what results', 'what results from', 'what comes', 'what comes next',
            'what leads', 'what leads to', 'what follows', 'what follows from',
        ]
        
        for pattern in causal_patterns:
            if pattern in task_lower:
                return True
        
        return False
    def _ensure_node_exists(self, node: str) -> None:
        
        if node not in self.causal_graph:
            self.causal_graph[node] = {}
        if node not in self.causal_graph_reverse:
            self.causal_graph_reverse[node] = []
        try:
            self._ensure_node_index(node)
        except Exception:
            pass

    def add_causal_relationship(
        self, 
        source: str, 
        target: str, 
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)

        meta = {
            "strength": float(strength),
            "relation_type": relation_type.value if isinstance(relation_type, Enum) else str(relation_type),
            "confidence": float(confidence),
        }

        self.causal_graph.setdefault(source, {})[target] = meta

        if source not in self.causal_graph_reverse.get(target, []):
            self.causal_graph_reverse.setdefault(target, []).append(source)

        try:
            u_idx = self._ensure_node_index(source)
            v_idx = self._ensure_node_index(target)
            try:
                existing = self._graph.get_edge_data(u_idx, v_idx)
            except Exception:
                existing = None

            if existing is None:
                try:
                    self._graph.add_edge(u_idx, v_idx, meta)
                except Exception:
                    try:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"rustworkx.add_edge failed for {source}->{target}; continuing with dict-only graph."
                        )
                    except Exception:
                        pass
            else:
                try:
                    if isinstance(existing, dict):
                        existing.update(meta)
                        try:
                            import logging
                            logging.getLogger(__name__).debug(
                                f"Updated rustworkx edge data for {source}->{target} in-place."
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            edge_idx = self._graph.get_edge_index(u_idx, v_idx)
                        except Exception:
                            edge_idx = None
                        if edge_idx is not None and edge_idx >= 0:
                            try:
                                self._graph.remove_edge(edge_idx)
                                self._graph.add_edge(u_idx, v_idx, meta)
                                try:
                                    import logging
                                    logging.getLogger(__name__).debug(
                                        f"Replaced rustworkx edge for {source}->{target} with updated metadata."
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    import logging
                                    logging.getLogger(__name__).warning(
                                        f"Could not replace rustworkx edge for {source}->{target}; keeping dict-only metadata."
                                    )
                                except Exception:
                                    pass
                        else:
                            try:
                                import logging
                                logging.getLogger(__name__).debug(
                                    f"rustworkx edge exists but index lookup failed for {source}->{target}; dict metadata used."
                                )
                            except Exception:
                                pass
                except Exception:
                    try:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Failed updating rustworkx edge for {source}->{target}; continuing with dict-only graph."
                        )
                    except Exception:
                        pass
        except Exception:
            try:
                import logging
                logging.getLogger(__name__).warning(
                    "rustworkx operation failed during add_causal_relationship; continuing with dict-only graph."
                )
            except Exception:
                pass
    
    def _get_parents(self, node: str) -> List[str]:
        
        return self.causal_graph_reverse.get(node, [])
    
    def _get_children(self, node: str) -> List[str]:
        
        return list(self.causal_graph.get(node, {}).keys())

    def _ensure_node_index(self, name: str) -> int:
        
        if name in self._node_to_index:
            return self._node_to_index[name]
        idx = self._graph.add_node(name)
        self._node_to_index[name] = idx
        self._index_to_node[idx] = name
        return idx

    def _node_index(self, name: str) -> Optional[int]:
        
        return self._node_to_index.get(name)

    def _node_name(self, idx: int) -> Optional[str]:
        
        return self._index_to_node.get(idx)

    def _edge_strength(self, source: str, target: str) -> float:
        
        edge = self.causal_graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0
    
    def _topological_sort(self) -> List[str]:
        
        try:
            order_idx = rx.topological_sort(self._graph)
            result = [self._node_name(i) for i in order_idx if self._node_name(i) is not None]
            for n in list(self.causal_graph.keys()):
                if n not in result:
                    result.append(n)
            return result
        except Exception:
            in_degree: Dict[str, int] = {node: 0 for node in self.causal_graph.keys()}
            for node in self.causal_graph:
                for child in self._get_children(node):
                    in_degree[child] = in_degree.get(child, 0) + 1
            
            queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
            result: List[str] = []
            while queue:
                node = queue.pop(0)
                result.append(node)
                for child in self._get_children(node):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            return result
    
    def identify_causal_chain(self, start: str, end: str) -> List[str]:
        
        if start not in self.causal_graph or end not in self.causal_graph:
            return []
        
        if start == end:
            return [start]
        
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: set = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            for child in self._get_children(current):
                if child == end:
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []  # No path found
    
    
    def _has_path(self, start: str, end: str) -> bool:
        
        if start == end:
            return True
        
        stack = [start]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for child in self._get_children(current):
                if child == end:
                    return True
                if child not in visited:
                    stack.append(child)
        
        return False

    def clear_cache(self) -> None:
        
        with self._prediction_cache_lock:
            self._prediction_cache.clear()
            self._prediction_cache_order.clear()

    def enable_cache(self, flag: bool) -> None:
        
        with self._prediction_cache_lock:
            self._cache_enabled = bool(flag)
    
    
    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        
        z: Dict[str, float] = {}
        for k, v in state.items():
            s = self.standardization_stats.get(k)
            if s and s.get("std", 0.0) > 0:
                z[k] = (v - s["mean"]) / s["std"]
            else:
                z[k] = v
        return z
    
    def _destandardize_value(self, var: str, z_value: float) -> float:
        
        s = self.standardization_stats.get(var)
        if s and s.get("std", 0.0) > 0:
            return z_value * s["std"] + s["mean"]
        return z_value
    
    def _predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        
        if self.use_nonlinear_scm:
            z_pred = self._predict_z(factual_state, interventions, use_noise=None)
            return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

        raw = factual_state.copy()
        raw.update(interventions)
        
        z_state = self._standardize_state(raw)
        z_pred = dict(z_state)
        
        for node in self._topological_sort():
            if node in interventions:
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue
            
            parents = self._get_parents(node)
            if not parents:
                continue
            
            s = 0.0
            for p in parents:
                pz = z_pred.get(p, z_state.get(p, 0.0))
                strength = self._edge_strength(p, node)
                s += pz * strength
            
            z_pred[node] = s
        
        return {v: self._destandardize_value(v, z) for v, z in z_pred.items()}

    def _predict_z(self, factual_state: Dict[str, float], interventions: Dict[str, float], use_noise: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        
        raw = factual_state.copy()
        raw.update(interventions)
        z_state = self._standardize_state(raw)
        z_pred: Dict[str, float] = dict(z_state)

        for node in self._topological_sort():
            if node in interventions:
                z_pred[node] = z_state.get(node, 0.0)
                continue

            parents = self._get_parents(node)
            if not parents:
                z_val = float(use_noise.get(node, 0.0)) if use_noise else z_state.get(node, 0.0)
                z_pred[node] = z_val
                continue

            linear_term = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_state.get(p, 0.0))
                beta = self._edge_strength(p, node)
                linear_term += parent_z * beta

            interaction_term = 0.0
            for (p1, p2) in self.interaction_terms.get(node, []):
                if p1 in parents and p2 in parents:
                    z1 = z_pred.get(p1, z_state.get(p1, 0.0))
                    z2 = z_pred.get(p2, z_state.get(p2, 0.0))
                    gamma = 0.0
                    edge_data = self.causal_graph.get(p1, {}).get(node, {})
                    if isinstance(edge_data, dict):
                        gamma = float(edge_data.get("interaction_strength", {}).get(p2, 0.0))
                    interaction_term += gamma * z1 * z2

            model_z = linear_term + interaction_term

            if use_noise:
                model_z += float(use_noise.get(node, 0.0))

            if self.nonlinear_activation == "tanh":
                model_z_act = float(np.tanh(model_z) * 3.0)  # scale to limit
            else:
                model_z_act = float(model_z)

            observed_z = z_state.get(node, 0.0)

            threshold = float(getattr(self, "shock_preserve_threshold", 1e-3))
            if abs(observed_z) > threshold:
                z_pred[node] = float(observed_z)
            else:
                z_pred[node] = float(model_z_act)

        return z_pred

    def aap(self, factual_state: Dict[str, float], interventions: Dict[str, float]) -> Dict[str, float]:
        
        return self.counterfactual_abduction_action_prediction(factual_state, interventions)

    def _predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        
        with self._prediction_cache_lock:
            cache_enabled = self._cache_enabled
        if not cache_enabled:
            return self._predict_outcomes(factual_state, interventions)

        state_key = tuple(sorted([(k, float(v)) for k, v in factual_state.items()]))
        inter_key = tuple(sorted([(k, float(v)) for k, v in interventions.items()]))
        cache_key = (state_key, inter_key)

        with self._prediction_cache_lock:
            if cache_key in self._prediction_cache:
                return dict(self._prediction_cache[cache_key])

        result = self._predict_outcomes(factual_state, interventions)

        with self._prediction_cache_lock:
            if len(self._prediction_cache_order) >= self._prediction_cache_max:
                remove_count = max(1, self._prediction_cache_max // 10)
                for _ in range(remove_count):
                    old = self._prediction_cache_order.pop(0)
                    if old in self._prediction_cache:
                        del self._prediction_cache[old]

            self._prediction_cache_order.append(cache_key)
            self._prediction_cache[cache_key] = dict(result)
        return result

    def _get_descendants(self, node: str) -> List[str]:
        
        if node not in self.causal_graph:
            return []
        stack = [node]
        visited = set()
        descendants: List[str] = []
        while stack:
            cur = stack.pop()
            for child in self._get_children(cur):
                if child in visited:
                    continue
                visited.add(child)
                descendants.append(child)
                stack.append(child)
        return descendants

    def counterfactual_abduction_action_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        
        z = self._standardize_state(factual_state)

        noise: Dict[str, float] = {}
        for node in self._topological_sort():
            parents = self._get_parents(node)
            if not parents:
                noise[node] = float(z.get(node, 0.0))
                continue

            pred_z = 0.0
            for p in parents:
                pz = z.get(p, 0.0)
                strength = self._edge_strength(p, node)
                pred_z += pz * strength

            noise[node] = float(z.get(node, 0.0) - pred_z)

        cf_raw = factual_state.copy()
        cf_raw.update(interventions)
        z_cf = self._standardize_state(cf_raw)

        z_pred: Dict[str, float] = {}
        for node in self._topological_sort():
            if node in interventions:
                z_pred[node] = float(z_cf.get(node, 0.0))
                continue

            parents = self._get_parents(node)
            if not parents:
                z_pred[node] = float(noise.get(node, 0.0))
                continue

            val = 0.0
            for p in parents:
                parent_z = z_pred.get(p, z_cf.get(p, 0.0))
                strength = self._edge_strength(p, node)
                val += parent_z * strength

            z_pred[node] = float(val + noise.get(node, 0.0))

        return {v: self._destandardize_value(v, z_val) for v, z_val in z_pred.items()}

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        
        def _ancestors(node: str) -> set:
            stack = [node]
            visited = set()
            while stack:
                cur = stack.pop()
                for p in self._get_parents(cur):
                    if p in visited:
                        continue
                    visited.add(p)
                    stack.append(p)
            return visited

        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        treat_anc = _ancestors(treatment)
        out_anc = _ancestors(outcome)
        common = treat_anc.intersection(out_anc)
        return list(common)

    def identify_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []

        parents_t = set(self._get_parents(treatment))
        descendants_t = set(self._get_descendants(treatment))
        adjustment = [z for z in parents_t if z not in descendants_t and z != outcome]
        return adjustment
    
    def _calculate_scenario_probability(
        self,
        factual_state: Dict[str, float], 
        interventions: Dict[str, float]
    ) -> float:
        
        z_sq = 0.0
        for var, new in interventions.items():
            s = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            mu, sd = s.get("mean", 0.0), s.get("std", 1.0) or 1.0
            old = factual_state.get(var, mu)
            dz = (new - mu) / sd - (old - mu) / sd
            z_sq += float(dz) * float(dz)
        
        p = 0.95 * float(np.exp(-0.5 * z_sq)) + 0.05
        return float(max(0.05, min(0.98, p)))
    
    def generate_counterfactual_scenarios(
        self,
        factual_state: Dict[str, float],
        target_variables: List[str],
        max_scenarios: int = 5
    ) -> List[CounterfactualScenario]:
        
        self.ensure_standardization_stats(factual_state)

        scenarios: List[CounterfactualScenario] = []
        z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        for i, tv in enumerate(target_variables[:max_scenarios]):
            stats = self.standardization_stats.get(tv, {"mean": 0.0, "std": 1.0})
            cur = factual_state.get(tv, stats.get("mean", 0.0))

            if not stats or stats.get("std", 0.0) <= 0:
                base = cur
                abs_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
                vals = [base + step for step in abs_steps]
            else:
                mean = stats["mean"]
                std = stats["std"]
                cz = (cur - mean) / std
                vals = [(cz + dz) * std + mean for dz in z_steps]

            for j, v in enumerate(vals):
                interventions = {tv: float(v)}
                scenarios.append(
                    CounterfactualScenario(
                        name=f"scenario_{i}_{j}",
                        interventions=interventions,
                        expected_outcomes=self._predict_outcomes(
                            factual_state, interventions
                        ),
                        probability=self._calculate_scenario_probability(
                            factual_state, interventions
                        ),
                        reasoning=f"Intervention on {tv} with value {v}",
                    )
                )

        return scenarios
    
    def analyze_causal_strength(self, source: str, target: str) -> Dict[str, float]:
        
        if source not in self.causal_graph or target not in self.causal_graph.get(source, {}):
            return {"strength": 0.0, "confidence": 0.0, "path_length": float('inf')}
        
        edge = self.causal_graph[source].get(target, {})
        strength = float(edge.get("strength", 0.0)) if isinstance(edge, dict) else float(edge)
        path = self.identify_causal_chain(source, target)
        path_length = len(path) - 1 if path else float('inf')
        
        return {
            "strength": float(strength),
            "confidence": float(edge.get("confidence", 1.0) if isinstance(edge, dict) else 1.0),
            "path_length": path_length,
            "relation_type": edge.get("relation_type", CausalRelationType.DIRECT.value) if isinstance(edge, dict) else CausalRelationType.DIRECT.value
        }
    
    def set_standardization_stats(
        self,
        variable: str,
        mean: float,
        std: float
    ) -> None:
        
        self.standardization_stats[variable] = {"mean": mean, "std": std if std > 0 else 1.0}
    
    def ensure_standardization_stats(self, state: Dict[str, float]) -> None:
        
        for var, val in state.items():
            if var not in self.standardization_stats:
                self.standardization_stats[var] = {"mean": float(val), "std": 1.0}
    
    def get_nodes(self) -> List[str]:
        
        return list(self.causal_graph.keys())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        
        edges = []
        for source, targets in self.causal_graph.items():
            for target in targets.keys():
                edges.append((source, target))
        return edges
    
    def is_dag(self) -> bool:
        
        try:
            return rx.is_directed_acyclic_graph(self._graph)
        except Exception:
            def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
                
                visited.add(node)
                rec_stack.add(node)
                
                for child in self._get_children(node):
                    if child not in visited:
                        if has_cycle(child, visited, rec_stack):
                            return True
                    elif child in rec_stack:
                        return True
                
                rec_stack.remove(node)
                return False
            
            visited = set()
            rec_stack = set()
            
            for node in self.causal_graph:
                if node not in visited:
                    if has_cycle(node, visited, rec_stack):
                        return False
            
            return True
    
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        streaming_callback: Optional[Any] = None,
        n: int = 1,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        """
        Run the agent with support for both standard Agent features and causal analysis.
        
        This method maintains compatibility with the parent Agent class while adding
        causal reasoning capabilities. It routes to causal analysis when appropriate,
        otherwise delegates to the parent Agent's standard functionality.
        
        Args:
            task: Task string for LLM analysis, or state dict for causal evolution
            img: Optional image path for vision tasks (delegates to parent)
            imgs: Optional list of images (delegates to parent)
            correct_answer: Optional correct answer for validation (delegates to parent)
            streaming_callback: Optional callback for streaming output (delegates to parent)
            n: Number of runs (delegates to parent)
            initial_state: Initial state dictionary for causal evolution
            target_variables: Target variables for counterfactual analysis
            max_steps: Maximum evolution steps for causal analysis
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with causal analysis results, or standard Agent output
        """
        # Check if this is a causal analysis operation
        # Criteria: initial_state or target_variables explicitly provided, or task is a dict,
        # OR automatic detection based on task content
        is_causal_operation = (
            initial_state is not None or
            target_variables is not None or
            (task is not None and isinstance(task, dict)) or
            (task is not None and isinstance(task, str) and task.strip().startswith('{')) or
            self._should_trigger_causal_analysis(task)  # Automatic detection
        )
        
        # Delegate to parent Agent for all standard operations
        # This includes: images, streaming, handoffs, multiple runs, regular text tasks, etc.
        # Only use causal analysis if explicitly requested via causal operation indicators
        if not is_causal_operation:
            # All standard Agent operations go to parent (handoffs, images, streaming, etc.)
            return super().run(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                streaming_callback=streaming_callback,
                n=n,
                *args,
                **kwargs,
            )
        
        # Causal analysis operations - only when explicitly indicated
        if task is not None and isinstance(task, str) and initial_state is None and not task.strip().startswith('{'):
            return self._run_llm_causal_analysis(task, **kwargs)
        
        if task is not None and initial_state is None:
            initial_state = task
        
        if not isinstance(initial_state, dict):
            try:
                import json
                parsed = json.loads(initial_state)
                if isinstance(parsed, dict):
                    initial_state = parsed
                else:
                    return {"error": "initial_state JSON must decode to a dict"}
            except Exception:
                return {"error": "initial_state must be a dict or JSON-encoded dict"}

        if target_variables is None:
            target_variables = list(self.causal_graph.keys())
        
        def _resolve_max_steps(value: Union[int, str]) -> int:
            if isinstance(value, str) and value == "auto":
                return max(1, len(self.causal_graph))
            try:
                return int(value)
            except Exception:
                return max(1, len(self.causal_graph))

        effective_steps = _resolve_max_steps(max_steps if max_steps != 1 or self.causal_max_loops == 1 else self.causal_max_loops)
        if max_steps == 1 and self.causal_max_loops != 1:
            effective_steps = _resolve_max_steps(self.causal_max_loops)

        current_state = initial_state.copy()
        for step in range(effective_steps):
            current_state = self._predict_outcomes(current_state, {})
        
        self.ensure_standardization_stats(current_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            current_state,
            target_variables,
            max_scenarios=5
        )
        
        return {
            "initial_state": initial_state,
            "evolved_state": current_state,
            "counterfactual_scenarios": counterfactual_scenarios,
            "causal_graph_info": {
                "nodes": self.get_nodes(),
                "edges": self.get_edges(),
                "is_dag": self.is_dag()
            },
            "steps": effective_steps
        }

    def _call_llm_directly(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LLM directly using litellm.completion(), bypassing Agent tool execution.
        
        This method extracts the model configuration from the Agent and makes a direct
        API call to get plain text/JSON responses without function calling.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt (defaults to Agent's system prompt)
            
        Returns:
            The raw text response from the LLM
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("litellm is required for direct LLM calls")
        
        # Get model configuration from Agent
        model_name = getattr(self, 'model_name', 'gpt-4o')
        api_key = getattr(self, 'llm_api_key', None) or os.getenv('OPENAI_API_KEY')
        base_url = getattr(self, 'llm_base_url', None)
        
        # Get system prompt
        if system_prompt is None:
            system_prompt = getattr(self, 'system_prompt', None)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call litellm directly
        try:
            response = litellm.completion(
                model=model_name,
                messages=messages,
                api_key=api_key,
                api_base=base_url,
                temperature=getattr(self, 'temperature', 0.5),
                max_tokens=getattr(self, 'max_tokens', 4096),
            )
            
            # Extract text from response
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content if content else ""
            else:
                logger.warning("Empty response from LLM")
                return ""
        except Exception as e:
            logger.error(f"Error calling LLM directly: {e}")
            raise
    
    def _extract_variables_ml_based(self, task: str) -> bool:
        """
        Extract variables using ML/NLP-based approach - use LLM to generate structured output,
        parse the function call from the response, and automatically invoke the handler.
        
        This bypasses unreliable function calling by directly parsing LLM output and
        invoking the handler programmatically.
        
        Args:
            task: The task string to analyze
            
        Returns:
            True if variables were successfully extracted, False otherwise
        """
        import json
        import re
        
        logger.info(f"Starting ML-based variable extraction for task: {task[:100]}...")
        
        # Use LLM to generate structured JSON output with variables and edges
        extraction_prompt = f"""Analyze this task and extract causal variables and relationships.

Task: {task}

Return a JSON object with this exact structure:
{{
    "required_variables": ["var1", "var2", "var3"],
    "causal_edges": [["var1", "var2"], ["var2", "var3"]],
    "reasoning": "Brief explanation of why these variables are needed",
    "optional_variables": ["var4"],
    "counterfactual_variables": ["var1", "var2"]
}}

Extract ALL relevant variables and causal relationships. Be thorough.
Return ONLY valid JSON, no other text."""

        try:
            # Call LLM directly (bypasses Agent tool execution)
            # This gives us pure text/JSON response without function calling
            # The LLM is still the core component - we just parse its output programmatically
            raw_response = self._call_llm_directly(extraction_prompt)
            
            extracted_data = None
            
            # Parse JSON from LLM text response
            # The LLM returns plain text with JSON embedded, so we extract it
            response_text = str(raw_response)
            
            # Try to extract JSON from response text
            json_match = re.search(r'\{[^{}]*"required_variables"[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                # Try to find any JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    # Try to fix common JSON issues
                    # Remove invalid escape sequences
                    json_str = json_str.replace('\\"', '"').replace("\\'", "'")
                    # Try parsing
                    extracted_data = json.loads(json_str)
                    logger.info("Parsed JSON from text response")
                except json.JSONDecodeError as e:
                    # Try to extract just the JSON object more carefully
                    try:
                        # Find the innermost complete JSON object
                        brace_count = 0
                        start_idx = json_str.find('{')
                        if start_idx >= 0:
                            for i in range(start_idx, len(json_str)):
                                if json_str[i] == '{':
                                    brace_count += 1
                                elif json_str[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = json_str[start_idx:i+1]
                                        extracted_data = json.loads(json_str)
                                        logger.info("Parsed JSON after fixing structure")
                                        break
                    except (json.JSONDecodeError, ValueError) as e2:
                        logger.warning(f"Failed to parse JSON after fixes: {e2}")
                        logger.debug(f"JSON string was: {json_str[:500]}")
            
            # Process extracted data
            if extracted_data:
                # Validate structure
                required_vars = extracted_data.get("required_variables", [])
                causal_edges = extracted_data.get("causal_edges", [])
                reasoning = extracted_data.get("reasoning", "Extracted from task analysis")
                optional_vars = extracted_data.get("optional_variables", [])
                counterfactual_vars = extracted_data.get("counterfactual_variables", [])
                
                if required_vars and causal_edges:
                    # Automatically invoke the handler with extracted data
                    logger.info(f"Extracted {len(required_vars)} variables, {len(causal_edges)} edges via ML")
                    result = self._extract_causal_variables_handler(
                        required_variables=required_vars,
                        causal_edges=causal_edges,
                        reasoning=reasoning,
                        optional_variables=optional_vars if optional_vars else None,
                        counterfactual_variables=counterfactual_vars if counterfactual_vars else None,
                    )
                    
                    # Check if extraction was successful
                    if result.get("status") == "success" and result.get("summary", {}).get("variables_added", 0) > 0:
                        logger.info(f"ML-based extraction successful: {result.get('summary')}")
                        return True
                    else:
                        logger.warning(f"ML-based extraction returned: {result}")
                else:
                    logger.warning(f"Extracted data missing required fields. required_variables: {required_vars}, causal_edges: {causal_edges}")
            else:
                logger.warning("Could not extract data from LLM response")
                logger.debug(f"Raw response type: {type(raw_response)}, value: {str(raw_response)[:500]}")
                
        except Exception as e:
            logger.error(f"Error in ML-based extraction: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return len(self.causal_graph) > 0

    def _generate_causal_analysis_ml_based(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Generate causal analysis using ML-based approach - use LLM to generate structured output,
        parse the function call from the response, and automatically invoke the handler.
        
        This bypasses unreliable function calling by directly parsing LLM output and
        invoking the handler programmatically.
        
        Args:
            task: The task string to analyze
            
        Returns:
            Dictionary with causal analysis results, or None if extraction failed
        """
        import json
        import re
        
        logger.info(f"Starting ML-based causal analysis generation for task: {task[:100]}...")
        
        # Build comprehensive causal analysis prompt
        causal_prompt = self._build_causal_prompt(task)
        
        # Add instruction for structured output
        analysis_prompt = f"""{causal_prompt}

CRITICAL: You must return a JSON object with this EXACT structure. Do not include any text before or after the JSON.

Required JSON format:
{{
    "causal_analysis": "Detailed analysis of causal relationships and mechanisms. This must be a comprehensive text explanation.",
    "intervention_planning": "Planned interventions to test causal hypotheses.",
    "counterfactual_scenarios": [
        {{
            "scenario_name": "Scenario 1",
            "interventions": {{"var1": 10, "var2": 20}},
            "expected_outcomes": {{"target_var": 30}},
            "reasoning": "Why this scenario is important..."
        }}
    ],
    "causal_strength_assessment": "Assessment of relationship strengths and confounders.",
    "optimal_solution": "Recommended solution based on analysis."
}}

IMPORTANT: 
- The "causal_analysis" field is REQUIRED and must contain detailed text analysis
- Return ONLY the JSON object, no markdown, no code blocks, no explanations
- Ensure all fields are present and properly formatted"""

        try:
            # Call LLM directly (bypasses Agent tool execution)
            # This gives us pure text/JSON response without function calling
            # The LLM is still the core component - we just parse its output programmatically
            raw_response = self._call_llm_directly(analysis_prompt)
            
            extracted_data = None
            
            # Parse JSON from LLM text response
            # The LLM returns plain text with JSON embedded, so we extract it
            response_text = str(raw_response)
            
            # First, try to extract JSON from markdown code blocks (```json ... ```)
            json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
                try:
                    extracted_data = json.loads(json_str)
                    logger.info("Parsed JSON from markdown code block")
                except json.JSONDecodeError:
                    pass
            
            # If not found in code block, try to extract JSON directly
            if not extracted_data:
                # Try to find JSON object with causal_analysis field
                json_match = re.search(r'\{[^{}]*"causal_analysis"[^{}]*\}', response_text, re.DOTALL)
                if not json_match:
                    # Try to find any complete JSON object (handle nested objects)
                    # This regex finds the outermost complete JSON object
                    brace_count = 0
                    start_idx = response_text.find('{')
                    if start_idx >= 0:
                        for i in range(start_idx, len(response_text)):
                            if response_text[i] == '{':
                                brace_count += 1
                            elif response_text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response_text[start_idx:i+1]
                                    try:
                                        # Try to fix common JSON issues
                                        json_str = json_str.replace('\\"', '"').replace("\\'", "'")
                                        extracted_data = json.loads(json_str)
                                        logger.info("Parsed JSON from text response")
                                        break
                                    except json.JSONDecodeError:
                                        # Try without the escape fixes
                                        try:
                                            extracted_data = json.loads(response_text[start_idx:i+1])
                                            logger.info("Parsed JSON after removing escape fixes")
                                            break
                                        except json.JSONDecodeError as e:
                                            logger.debug(f"Failed to parse JSON: {e}")
                                            break
            
            # Process extracted data
            if extracted_data:
                # Validate structure - try multiple possible field names
                causal_analysis = (
                    extracted_data.get("causal_analysis", "") or
                    extracted_data.get("analysis", "") or
                    extracted_data.get("causal_analysis_text", "") or
                    str(extracted_data.get("analysis_text", ""))
                )
                intervention_planning = extracted_data.get("intervention_planning", "") or extracted_data.get("interventions", "")
                counterfactual_scenarios = extracted_data.get("counterfactual_scenarios", []) or extracted_data.get("scenarios", [])
                causal_strength_assessment = extracted_data.get("causal_strength_assessment", "") or extracted_data.get("strength_assessment", "")
                optimal_solution = extracted_data.get("optimal_solution", "") or extracted_data.get("solution", "")
                
                # Log what we extracted for debugging (use INFO so it's visible)
                logger.info(f"Extracted fields - causal_analysis: {bool(causal_analysis)}, scenarios: {len(counterfactual_scenarios)}")
                logger.info(f"Extracted data keys: {list(extracted_data.keys())}")
                if not causal_analysis:
                    # Log the full structure to see what we got
                    logger.info(f"Full extracted data (first 1000 chars): {str(extracted_data)[:1000]}")
                
                if causal_analysis:
                    # Automatically invoke the handler with extracted data
                    logger.info(f"Extracted causal analysis via ML ({len(causal_analysis)} chars, {len(counterfactual_scenarios)} scenarios)")
                    result = self._generate_causal_analysis_handler(
                        causal_analysis=causal_analysis,
                        intervention_planning=intervention_planning,
                        counterfactual_scenarios=counterfactual_scenarios,
                        causal_strength_assessment=causal_strength_assessment,
                        optimal_solution=optimal_solution,
                    )
                    
                    # Check if analysis was successful
                    if result.get("status") == "success":
                        logger.info(f"ML-based causal analysis generation successful")
                        # Return the analysis data for use in final result
                        return {
                            'causal_analysis': causal_analysis,
                            'intervention_planning': intervention_planning,
                            'counterfactual_scenarios': counterfactual_scenarios,
                            'causal_strength_assessment': causal_strength_assessment,
                            'optimal_solution': optimal_solution,
                        }
                    else:
                        logger.warning(f"ML-based analysis returned: {result}")
                else:
                    logger.warning("Extracted data missing causal_analysis field")
                    logger.info(f"Extracted data structure: {extracted_data}")
                    logger.info(f"Available keys: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else 'Not a dict'}")
                    # Try to use the raw response text as causal_analysis if JSON parsing failed
                    if not extracted_data or not isinstance(extracted_data, dict):
                        logger.info("Attempting to use raw response as causal analysis")
                        # Use the raw response as a fallback
                        causal_analysis = response_text[:5000]  # Limit length
                        if causal_analysis:
                            result = self._generate_causal_analysis_handler(
                                causal_analysis=causal_analysis,
                                intervention_planning="",
                                counterfactual_scenarios=[],
                                causal_strength_assessment="",
                                optimal_solution="",
                            )
                            if result.get("status") == "success":
                                return {
                                    'causal_analysis': causal_analysis,
                                    'intervention_planning': "",
                                    'counterfactual_scenarios': [],
                                    'causal_strength_assessment': "",
                                    'optimal_solution': "",
                                }
            else:
                logger.warning("Could not extract data from LLM response for causal analysis")
                logger.debug(f"Raw response type: {type(raw_response)}, value: {str(raw_response)[:500]}")
                
        except Exception as e:
            logger.error(f"Error in ML-based causal analysis generation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return None

    def _extract_variables_from_task(self, task: str) -> bool:
        """
        Extract variables and causal relationships from a task.
        
        Uses ML-based extraction (structured LLM output + automatic handler invocation)
        instead of relying on unreliable function calling.
        
        Args:
            task: The task string to analyze
            
        Returns:
            True if variables were successfully extracted, False otherwise
        """
        # Use ML-based extraction (more reliable than function calling)
        return self._extract_variables_ml_based(task)

    def _run_llm_causal_analysis(self, task: str, target_variables: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Run LLM-based causal analysis on a task.
        
        Args:
            task: Task string to analyze
            target_variables: Optional list of target variables for counterfactual scenarios.
                            If None, uses all variables in the causal graph.
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with causal analysis results
        """
        self.causal_memory = []
        
        # Check if causal graph is empty - if so, trigger variable extraction
        if len(self.causal_graph) == 0:
            logger.info("Causal graph is empty, starting variable extraction phase...")
            extraction_success = self._extract_variables_from_task(task)
            
            if not extraction_success:
                # Extraction failed, return error
                return {
                    'task': task,
                    'error': 'Variable extraction failed',
                    'message': (
                        'Could not extract variables from the task. '
                        'Please ensure the task describes causal relationships or variables, '
                        'or manually initialize the agent with variables and causal_edges.'
                    ),
                    'causal_graph_info': {
                        'nodes': [],
                        'edges': [],
                        'is_dag': True
                    }
                }
            
            logger.info(
                f"Variable extraction completed. "
                f"Graph now has {len(self.causal_graph)} variables and "
                f"{sum(len(children) for children in self.causal_graph.values())} edges."
            )
        
        # Use Agent's normal run method to get rich output and proper loop handling
        # This will show the LLM's rich output and respect max_loops
        causal_prompt = self._build_causal_prompt(task)
        
        # Store the current memory size before running to detect new entries
        memory_size_before = len(self.causal_memory)
        
        # Run the agent - this will execute tools and store results in causal_memory
        super().run(task=causal_prompt)
        
        # Extract the causal analysis from causal_memory (stored by tool handler)
        # Look for the most recent analysis entry
        analysis_result = None
        final_analysis = ""
        
        # Search backwards through causal_memory for the most recent analysis
        # Check both new entries (after memory_size_before) and all entries
        for entry in reversed(self.causal_memory):
            if isinstance(entry, dict):
                # Check if this is an analysis entry (has 'type' == 'analysis' or has 'causal_analysis' field)
                entry_type = entry.get('type')
                has_causal_analysis = 'causal_analysis' in entry
                
                if entry_type == 'analysis' or has_causal_analysis:
                    analysis_result = entry
                    final_analysis = entry.get('causal_analysis', '')
                    if final_analysis:
                        logger.info(f"Found causal analysis in memory (type: {entry_type}, length: {len(final_analysis)})")
                        break
        
        # If no analysis found in memory, use ML-based generation as fallback
        if not final_analysis:
            logger.warning("No causal analysis found in causal_memory, using ML-based generation fallback")
            ml_analysis = self._generate_causal_analysis_ml_based(task)
            if ml_analysis and ml_analysis.get('causal_analysis'):
                # Use the returned analysis directly
                final_analysis = ml_analysis.get('causal_analysis', '')
                analysis_result = {
                    'type': 'analysis',
                    'causal_analysis': final_analysis,
                    'intervention_planning': ml_analysis.get('intervention_planning', ''),
                    'counterfactual_scenarios': ml_analysis.get('counterfactual_scenarios', []),
                    'causal_strength_assessment': ml_analysis.get('causal_strength_assessment', ''),
                    'optimal_solution': ml_analysis.get('optimal_solution', ''),
                }
                # Also ensure it's stored in memory for consistency
                if analysis_result not in self.causal_memory:
                    self.causal_memory.append(analysis_result)
                if final_analysis:
                    logger.info(f"Found causal analysis from ML-based fallback (length: {len(final_analysis)})")
        
        # If still no analysis found, use the LLM's response as final fallback
        if not final_analysis:
            logger.warning("No causal analysis found in causal_memory, attempting fallback from conversation history")
            # Try to get the last response from the conversation
            if hasattr(self, 'short_memory') and self.short_memory:
                # Conversation object has conversation_history attribute, not get_messages()
                if hasattr(self.short_memory, 'conversation_history'):
                    conversation_history = self.short_memory.conversation_history
                    if conversation_history:
                        # Get the last assistant message
                        for msg in reversed(conversation_history):
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                final_analysis = msg.get('content', '')
                                if final_analysis:
                                    logger.info(f"Extracted causal analysis from conversation history (length: {len(final_analysis)})")
                                break
                            elif hasattr(msg, 'role') and msg.role == 'assistant':
                                final_analysis = getattr(msg, 'content', str(msg))
                                if final_analysis:
                                    logger.info(f"Extracted causal analysis from conversation history (length: {len(final_analysis)})")
                                break
        
        if not final_analysis:
            logger.warning(f"Could not extract causal analysis. Memory size: {len(self.causal_memory)} (was {memory_size_before})")
        
        default_state = {var: 0.0 for var in self.get_nodes()}
        self.ensure_standardization_stats(default_state)
        
        # Use provided target_variables or default to all variables
        if target_variables is None:
            target_variables = self.get_nodes()
        
        # Limit to top variables if too many
        target_vars = target_variables[:5] if len(target_variables) > 5 else target_variables
        
        # Use counterfactual scenarios from ML analysis if available
        if analysis_result and isinstance(analysis_result, dict):
            ml_scenarios = analysis_result.get('counterfactual_scenarios', [])
            if ml_scenarios:
                counterfactual_scenarios = ml_scenarios
            else:
                # Fallback to generated scenarios
                counterfactual_scenarios = self.generate_counterfactual_scenarios(
                    default_state,
                    target_vars,
                    max_scenarios=5
                )
        else:
            # Fallback to generated scenarios
            counterfactual_scenarios = self.generate_counterfactual_scenarios(
                default_state,
                target_vars,
            max_scenarios=5
        )
        
        return {
            'task': task,
            'causal_analysis': final_analysis,
            'counterfactual_scenarios': counterfactual_scenarios,
            'causal_graph_info': {
                'nodes': self.get_nodes(),
                'edges': self.get_edges(),
                'is_dag': self.is_dag()
            },
            'analysis_steps': self.causal_memory,
            'intervention_planning': analysis_result.get('intervention_planning', '') if analysis_result else '',
            'causal_strength_assessment': analysis_result.get('causal_strength_assessment', '') if analysis_result else '',
            'optimal_solution': analysis_result.get('optimal_solution', '') if analysis_result else '',
        }

    # =========================
    # Compatibility extensions
    # =========================

    # ---- Helpers ----
    @staticmethod
    def _require_pandas() -> None:
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for this operation. Install pandas to proceed.")

    @staticmethod
    def _require_scipy() -> None:
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for this operation. Install scipy to proceed.")

    @staticmethod
    def _require_cvxpy() -> None:
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for this operation. Install cvxpy to proceed.")

    def _edge_strength(self, source: str, target: str) -> float:
        edge = self.causal_graph.get(source, {}).get(target, None)
        if isinstance(edge, dict):
            return float(edge.get("strength", 0.0))
        try:
            return float(edge) if edge is not None else 0.0
        except Exception:
            return 0.0

    class _TimeDebug:
        def __init__(self, name: str) -> None:
            self.name = name
            self.start = 0.0
        def __enter__(self):
            try:
                import time
                self.start = time.perf_counter()
            except Exception:
                self.start = 0.0
            return self
        def __exit__(self, exc_type, exc, tb):
            if logger.isEnabledFor(logging.DEBUG) and self.start:
                try:
                    import time
                    duration = time.perf_counter() - self.start
                    logger.debug(f"{self.name} completed in {duration:.4f}s")
                except Exception:
                    pass

    def _ensure_edge(self, source: str, target: str) -> None:
        """Ensure edge exists in both dict graph and rustworkx graph."""
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)
        if target not in self.causal_graph.get(source, {}):
            self.causal_graph.setdefault(source, {})[target] = {"strength": 0.0, "confidence": 1.0}
            try:
                u_idx = self._ensure_node_index(source)
                v_idx = self._ensure_node_index(target)
                if self._graph.get_edge_data(u_idx, v_idx) is None:
                    self._graph.add_edge(u_idx, v_idx, self.causal_graph[source][target])
            except Exception:
                pass

    # Convenience graph-like methods for compatibility
    def add_nodes_from(self, nodes: List[str]) -> None:
        for n in nodes:
            self._ensure_node_exists(n)

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> None:
        for u, v in edges:
            self.add_causal_relationship(u, v)

    def edges(self) -> List[Tuple[str, str]]:
        return self.get_edges()

    # ---- Batched predictions ----
    def _predict_outcomes_batch(
        self,
        factual_states: List[Dict[str, float]],
        interventions: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
    ) -> List[Dict[str, float]]:
        """
        Batched deterministic SCM forward pass. Uses shared topology and vectorized parent aggregation.
        """
        if not factual_states:
            return []
        if len(factual_states) == 1 or not self.enable_batch_predict:
            return [self._predict_outcomes(factual_states[0], interventions if isinstance(interventions, dict) else (interventions or {}))]

        batch = len(factual_states)
        if interventions is None:
            interventions_list = [{} for _ in range(batch)]
        elif isinstance(interventions, list):
            interventions_list = interventions
        else:
            interventions_list = [interventions for _ in range(batch)]

        topo = self._topological_sort()
        parents_map = {node: self._get_parents(node) for node in topo}
        stats = self.standardization_stats
        z_pred: Dict[str, np.ndarray] = {}

        # Initialize z with raw + interventions standardized
        for node in topo:
            arr = np.empty(batch, dtype=float)
            mean = stats.get(node, {}).get("mean", 0.0)
            std = stats.get(node, {}).get("std", 1.0) or 1.0
            for i in range(batch):
                raw_val = interventions_list[i].get(node, factual_states[i].get(node, 0.0))
                arr[i] = (raw_val - mean) / std
            z_pred[node] = arr

        # Propagate for non-intervened nodes
        for node in topo:
            parents = parents_map.get(node, [])
            if not parents:
                continue
            arr = z_pred[node]
            # Only recompute if node not directly intervened
            intervene_mask = np.array([node in interventions_list[i] for i in range(batch)], dtype=bool)
            if np.all(intervene_mask):
                continue
            if not parents:
                continue
            parent_matrix = np.vstack([z_pred[p] for p in parents])  # shape (k, batch)
            strengths = np.array([self._edge_strength(p, node) for p in parents], dtype=float).reshape(-1, 1)
            combined = (strengths * parent_matrix).sum(axis=0)
            if intervene_mask.any():
                # preserve intervened samples
                arr = np.where(intervene_mask, arr, combined)
            else:
                arr = combined
            z_pred[node] = arr

        # De-standardize
        outputs: List[Dict[str, float]] = []
        for i in range(batch):
            out: Dict[str, float] = {}
            for node, z_arr in z_pred.items():
                s = stats.get(node, {"mean": 0.0, "std": 1.0})
                out[node] = float(z_arr[i] * s.get("std", 1.0) + s.get("mean", 0.0))
            outputs.append(out)
        return outputs

    # Convenience graph-like methods for compatibility
    def add_nodes_from(self, nodes: List[str]) -> None:
        for n in nodes:
            self._ensure_node_exists(n)

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> None:
        for u, v in edges:
            self.add_causal_relationship(u, v)

    def edges(self) -> List[Tuple[str, str]]:
        return self.get_edges()

    # ---- Data-driven fitting ----
    def fit_from_dataframe(
        self,
        df: Any,
        variables: List[str],
        window: int = 30,
        decay_alpha: float = 0.9,
        ridge_lambda: float = 0.0,
        enforce_signs: bool = True
    ) -> None:
        """
        Fit edge strengths and standardization stats from a rolling window with recency weighting.
        """
        with self._TimeDebug("fit_from_dataframe"):
            self._require_pandas()
            if df is None:
                return
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
            if not variables:
                return
            missing = [v for v in variables if v not in df.columns]
            if missing:
                raise ValueError(f"Variables not in DataFrame: {missing}")
            window = max(1, int(window))
            if not (0 < decay_alpha <= 1):
                raise ValueError("decay_alpha must be in (0,1]")

            df_local = df[variables].dropna().copy()
            if df_local.empty:
                return
            window_df = df_local.tail(window)
            n = len(window_df)
            weights = np.array([decay_alpha ** (n - 1 - i) for i in range(n)], dtype=float)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

            # Standardization stats
            self.standardization_stats = {}
            for v in variables:
                m = float(window_df[v].mean())
                s = float(window_df[v].std(ddof=0))
                if s == 0:
                    s = 1.0
                self.standardization_stats[v] = {"mean": m, "std": s}
            for node in self.causal_graph.keys():
                if node not in self.standardization_stats:
                    self.standardization_stats[node] = {"mean": 0.0, "std": 1.0}

            # Estimate edge strengths
            for child in list(self.causal_graph.keys()):
                parents = self._get_parents(child)
                if not parents:
                    continue
                if child not in window_df.columns:
                    continue
                parent_vals = []
                for p in parents:
                    if p in window_df.columns:
                        stats = self.standardization_stats.get(p, {"mean": 0.0, "std": 1.0})
                        parent_vals.append(((window_df[p] - stats["mean"]) / stats["std"]).values)
                if not parent_vals:
                    continue
                X = np.vstack(parent_vals).T
                y_stats = self.standardization_stats.get(child, {"mean": 0.0, "std": 1.0})
                y = ((window_df[child] - y_stats["mean"]) / y_stats["std"]).values
                W = np.diag(weights)
                XtW = X.T @ W
                XtWX = XtW @ X
                if ridge_lambda > 0 and XtWX.size > 0:
                    k = XtWX.shape[0]
                    XtWX = XtWX + ridge_lambda * np.eye(k)
                try:
                    XtWX_inv = np.linalg.pinv(XtWX)
                    beta = XtWX_inv @ (XtW @ y)
                except Exception:
                    beta = np.zeros(X.shape[1])
                beta = np.asarray(beta)
                for idx, p in enumerate(parents):
                    strength = float(beta[idx]) if idx < len(beta) else 0.0
                    if enforce_signs:
                        sign = self.edge_sign_constraints.get((p, child))
                        if sign == 1 and strength < 0:
                            strength = 0.0
                        elif sign == -1 and strength > 0:
                            strength = 0.0
                    self._ensure_edge(p, child)
                    self.causal_graph[p][child]["strength"] = strength
                    self.causal_graph[p][child]["confidence"] = 1.0

    # ---- Uncertainty ----
    def quantify_uncertainty(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        with self._TimeDebug("quantify_uncertainty"):
            self._require_pandas()
            if df is None or not isinstance(df, pd.DataFrame):
                return {"edge_cis": {}, "samples": 0}
            usable = df[variables].dropna()
            if len(usable) < 10:
                return {"edge_cis": {}, "samples": 0}
            windows = max(1, int(windows))
            samples: Dict[Tuple[str, str], List[float]] = {}

        # Snapshot current strengths to restore later
        baseline_strengths: Dict[Tuple[str, str], float] = {}
        for u, targets in self.causal_graph.items():
            for v, meta in targets.items():
                try:
                    baseline_strengths[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                except Exception:
                    baseline_strengths[(u, v)] = 0.0
        baseline_stats = dict(self.standardization_stats)

        def _snapshot_strengths() -> Dict[Tuple[str, str], float]:
            snap: Dict[Tuple[str, str], float] = {}
            for u, targets in self.causal_graph.items():
                for v, meta in targets.items():
                    try:
                        snap[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        snap[(u, v)] = 0.0
            return snap

        def _bootstrap_single(df_sample: "pd.DataFrame") -> Dict[Tuple[str, str], float]:
            # Use a shallow clone to avoid mutating main agent when running in parallel
            clone = CRCAAgent(
                variables=list(self.causal_graph.keys()),
                causal_edges=self.get_edges(),
                model_name=self.model_name,
                max_loops=self.causal_max_loops,
                enable_batch_predict=self.enable_batch_predict,
                max_batch_size=self.max_batch_size,
                bootstrap_workers=0,
                use_async=self.use_async,
                seed=self.seed,
            )
            clone.edge_sign_constraints = dict(self.edge_sign_constraints)
            clone.standardization_stats = dict(baseline_stats)
            try:
                clone.fit_from_dataframe(
                    df=df_sample,
                    variables=variables,
                    window=min(30, len(df_sample)),
                    decay_alpha=0.9,
                    ridge_lambda=0.0,
                    enforce_signs=True,
                )
                return _snapshot_strengths_from_graph(clone.causal_graph)
            except Exception:
                return {}

        def _snapshot_strengths_from_graph(graph: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
            res: Dict[Tuple[str, str], float] = {}
            for u, targets in graph.items():
                for v, meta in targets.items():
                    try:
                        res[(u, v)] = float(meta.get("strength", 0.0)) if isinstance(meta, dict) else float(meta)
                    except Exception:
                        res[(u, v)] = 0.0
            return res

        use_parallel = self.bootstrap_workers > 0
        if use_parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.bootstrap_workers) as executor:
                futures = []
                for i in range(windows):
                    boot_df = usable.sample(n=len(usable), replace=True, random_state=self.seed + i)
                    futures.append(executor.submit(_bootstrap_single, boot_df))
                for fut in futures:
                    try:
                        res_strengths = fut.result()
                        for (u, v), w in res_strengths.items():
                            samples.setdefault((u, v), []).append(w)
                    except Exception:
                        continue
        else:
            for i in range(windows):
                boot_df = usable.sample(n=len(usable), replace=True, random_state=self.seed + i)
                try:
                    self.fit_from_dataframe(
                        df=boot_df,
                        variables=variables,
                        window=min(30, len(boot_df)),
                        decay_alpha=0.9,
                        ridge_lambda=0.0,
                        enforce_signs=True,
                    )
                    for (u, v), w in _snapshot_strengths().items():
                        samples.setdefault((u, v), []).append(w)
                except Exception:
                    continue

        # Restore baseline strengths and stats
        for (u, v), w in baseline_strengths.items():
            if u in self.causal_graph and v in self.causal_graph[u]:
                self.causal_graph[u][v]["strength"] = w
        self.standardization_stats = baseline_stats

        edge_cis: Dict[str, Tuple[float, float]] = {}
        for (u, v), arr in samples.items():
            arr_np = np.array(arr)
            lo = float(np.quantile(arr_np, (1 - alpha) / 2))
            hi = float(np.quantile(arr_np, 1 - (1 - alpha) / 2))
            edge_cis[f"{u}->{v}"] = (lo, hi)
        return {"edge_cis": edge_cis, "samples": windows}

    # ---- Optimization ----
    def gradient_based_intervention_optimization(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, Any]:
        self._require_scipy()
        from scipy.optimize import minimize  # type: ignore

        if not intervention_vars:
            return {"error": "intervention_vars cannot be empty", "optimal_intervention": {}, "success": False}

        bounds = []
        x0 = []
        for var in intervention_vars:
            cur = float(initial_state.get(var, 0.0))
            x0.append(cur)
            if constraints and var in constraints:
                bounds.append(constraints[var])
            else:
                bounds.append((cur - 3.0, cur + 3.0))

        def objective(x: np.ndarray) -> float:
            intervention = {intervention_vars[i]: float(x[i]) for i in range(len(x))}
            outcome = self._predict_outcomes(initial_state, intervention)
            return -float(outcome.get(target, 0.0))

        try:
            result = minimize(
                objective,
                x0=np.array(x0, dtype=float),
                method=method,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-6},
            )
            optimal_intervention = {intervention_vars[i]: float(result.x[i]) for i in range(len(result.x))}
            optimal_outcome = self._predict_outcomes(initial_state, optimal_intervention)
            return {
                "optimal_intervention": optimal_intervention,
                "optimal_target_value": float(optimal_outcome.get(target, 0.0)),
                "objective_value": float(result.fun),
                "success": bool(result.success),
                "iterations": int(getattr(result, "nit", 0)),
                "convergence_message": str(result.message),
            }
        except Exception as e:
            logger.debug(f"gradient_based_intervention_optimization failed: {e}")
            return {"error": str(e), "optimal_intervention": {}, "success": False}

    def bellman_optimal_intervention(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        horizon: int = 5,
        discount: float = 0.9,
    ) -> Dict[str, Any]:
        if not intervention_vars:
            return {"error": "intervention_vars cannot be empty"}
        horizon = max(1, int(horizon))
        rng = np.random.default_rng(self.seed)
        current_state = dict(initial_state)
        sequence: List[Dict[str, float]] = []

        def reward(state: Dict[str, float]) -> float:
            return float(state.get(target, 0.0))

        for _ in range(horizon):
            best_value = float("-inf")
            best_intervention: Dict[str, float] = {}
            for _ in range(10):
                candidate = {}
                for var in intervention_vars:
                    stats = self.standardization_stats.get(var, {"mean": current_state.get(var, 0.0), "std": 1.0})
                    candidate[var] = float(rng.normal(stats["mean"], stats["std"]))
                next_state = self._predict_outcomes(current_state, candidate)
                val = reward(next_state)
                if val > best_value:
                    best_value = val
                    best_intervention = candidate
            if best_intervention:
                sequence.append(best_intervention)
                current_state = self._predict_outcomes(current_state, best_intervention)

        return {
            "optimal_sequence": sequence,
            "final_state": current_state,
            "total_value": float(current_state.get(target, 0.0)),
            "horizon": horizon,
            "discount_factor": float(discount),
        }

    # ---- Time-series & causality ----
    def granger_causality_test(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data or variables"}
        data = df[[var1, var2]].dropna()
        if len(data) < max_lag * 2 + 5:
            return {"error": "Insufficient data"}
        try:
            from scipy.stats import f as f_dist  # type: ignore
        except Exception:
            return {"error": "scipy f distribution not available"}

        n = len(data)
        y = data[var2].values
        Xr = []
        Xu = []
        for t in range(max_lag, n):
            y_t = y[t]
            lags_var2 = [data[var2].iloc[t - i] for i in range(1, max_lag + 1)]
            lags_var1 = [data[var1].iloc[t - i] for i in range(1, max_lag + 1)]
            Xr.append(lags_var2)
            Xu.append(lags_var2 + lags_var1)
        y_vec = np.array(y[max_lag:], dtype=float)
        Xr = np.array(Xr, dtype=float)
        Xu = np.array(Xu, dtype=float)

        def ols(X: np.ndarray, yv: np.ndarray) -> Tuple[np.ndarray, float]:
            beta = np.linalg.pinv(X) @ yv
            y_pred = X @ beta
            rss = float(np.sum((yv - y_pred) ** 2))
            return beta, rss

        try:
            _, rss_r = ols(Xr, y_vec)
            _, rss_u = ols(Xu, y_vec)
            m = max_lag
            df2 = len(y_vec) - 2 * m - 1
            if df2 <= 0 or rss_u <= 1e-12:
                return {"error": "Degenerate case in F-test"}
            f_stat = ((rss_r - rss_u) / m) / (rss_u / df2)
            p_value = float(1.0 - f_dist.cdf(f_stat, m, df2))
            return {
                "f_statistic": float(f_stat),
                "p_value": p_value,
                "granger_causes": p_value < 0.05,
                "max_lag": max_lag,
                "restricted_rss": rss_r,
                "unrestricted_rss": rss_u,
            }
        except Exception as e:
            return {"error": str(e)}

    def vector_autoregression_estimation(
        self,
        df: Any,
        variables: List[str],
        max_lag: int = 2,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        data = df[variables].dropna()
        if len(data) < max_lag * len(variables) + 5:
            return {"error": "Insufficient data"}
        n_vars = len(variables)
        X_lag = []
        y_mat = []
        for t in range(max_lag, len(data)):
            y_row = [data[var].iloc[t] for var in variables]
            y_mat.append(y_row)
            lag_row = []
            for lag in range(1, max_lag + 1):
                for var in variables:
                    lag_row.append(data[var].iloc[t - lag])
            X_lag.append(lag_row)
        X = np.array(X_lag, dtype=float)
        Y = np.array(y_mat, dtype=float)
        coefficients: Dict[str, Any] = {}
        residuals = []
        for idx, var in enumerate(variables):
            y_vec = Y[:, idx]
            beta = np.linalg.pinv(X) @ y_vec
            y_pred = X @ beta
            res = y_vec - y_pred
            residuals.append(res)
            coefficients[var] = {"coefficients": beta.tolist()}
        residuals = np.array(residuals).T
        return {
            "coefficient_matrices": coefficients,
            "residuals": residuals.tolist(),
            "n_observations": len(Y),
            "n_variables": n_vars,
            "max_lag": max_lag,
            "variables": variables,
        }

    def compute_information_theoretic_measures(
        self,
        df: Any,
        variables: List[str],
    ) -> Dict[str, Any]:
        """
        Compute simple entropy and mutual information estimates using histograms.
        """
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        data = df[variables].dropna()
        if len(data) < 10:
            return {"error": "Insufficient data"}

        results: Dict[str, Any] = {"entropies": {}, "mutual_information": {}}
        for var in variables:
            if var not in data.columns:
                continue
            series = data[var].dropna()
            if len(series) < 5:
                continue
            n_bins = min(20, max(5, int(np.sqrt(len(series)))))
            hist, _ = np.histogram(series, bins=n_bins)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            results["entropies"][var] = float(entropy)

        # Pairwise mutual information
        for i, var1 in enumerate(variables):
            if var1 not in results["entropies"]:
                continue
            for var2 in variables[i + 1:]:
                if var2 not in results["entropies"]:
                    continue
                joint = data[[var1, var2]].dropna()
                if len(joint) < 5:
                    continue
                n_bins = min(10, max(3, int(np.cbrt(len(joint)))))
                hist2d, _, _ = np.histogram2d(joint[var1], joint[var2], bins=n_bins)
                hist2d = hist2d[hist2d > 0]
                probs_joint = hist2d / hist2d.sum()
                h_joint = -np.sum(probs_joint * np.log2(probs_joint))
                mi = results["entropies"][var1] + results["entropies"][var2] - float(h_joint)
                results["mutual_information"][f"{var1};{var2}"] = float(max(0.0, mi))

        return results

    # ---- Bayesian & attribution ----
    def bayesian_edge_inference(
        self,
        df: Any,
        parent: str,
        child: str,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        self._require_pandas()
        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "Invalid data"}
        if parent not in df.columns or child not in df.columns:
            return {"error": "Variables not found"}
        data = df[[parent, child]].dropna()
        if len(data) < 5:
            return {"error": "Insufficient data"}
        X = data[parent].values.reshape(-1, 1)
        y = data[child].values
        X_mean, X_std = X.mean(), X.std() or 1.0
        y_mean, y_std = y.mean(), y.std() or 1.0
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm
        beta_ols = float((np.linalg.pinv(XtX) @ Xty)[0])
        residuals = y_norm - X_norm @ np.array([beta_ols])
        sigma_sq = float(np.var(residuals))
        tau_likelihood = 1.0 / (sigma_sq + 1e-6)
        tau_prior = 1.0 / (prior_sigma ** 2)
        tau_post = tau_prior + tau_likelihood * len(data)
        mu_post = (tau_prior * prior_mu + tau_likelihood * len(data) * beta_ols) / tau_post
        sigma_post = math.sqrt(1.0 / tau_post)
        ci_lower = mu_post - 1.96 * sigma_post
        ci_upper = mu_post + 1.96 * sigma_post
        self.bayesian_priors[(parent, child)] = {"mu": prior_mu, "sigma": prior_sigma}
        return {
            "posterior_mean": float(mu_post),
            "posterior_std": float(sigma_post),
            "posterior_variance": float(sigma_post ** 2),
            "credible_interval_95": (float(ci_lower), float(ci_upper)),
            "ols_estimate": float(beta_ols),
            "prior_mu": float(prior_mu),
            "prior_sigma": float(prior_sigma),
        }

    def sensitivity_analysis(
        self,
        intervention: Dict[str, float],
        target: str,
        perturbation_size: float = 0.01,
    ) -> Dict[str, Any]:
        base_outcome = self._predict_outcomes({}, intervention)
        base_target = base_outcome.get(target, 0.0)
        sensitivities: Dict[str, float] = {}
        elasticities: Dict[str, float] = {}
        for var, val in intervention.items():
            perturbed = dict(intervention)
            perturbed[var] = val + perturbation_size
            perturbed_outcome = self._predict_outcomes({}, perturbed)
            pert_target = perturbed_outcome.get(target, 0.0)
            sensitivity = (pert_target - base_target) / perturbation_size
            sensitivities[var] = float(sensitivity)
            if abs(base_target) > 1e-6 and abs(val) > 1e-6:
                elasticities[var] = float(sensitivity * (val / base_target))
            else:
                elasticities[var] = 0.0
        most_inf = max(sensitivities.items(), key=lambda x: abs(x[1])) if sensitivities else (None, 0.0)
        total_sens = float(np.linalg.norm(list(sensitivities.values()))) if sensitivities else 0.0
        return {
            "sensitivities": sensitivities,
            "elasticities": elasticities,
            "total_sensitivity": total_sens,
            "most_influential_variable": most_inf[0],
            "most_influential_sensitivity": float(most_inf[1]),
        }

    def deep_root_cause_analysis(
        self,
        problem_variable: str,
        max_depth: int = 20,
        min_path_strength: float = 0.01,
    ) -> Dict[str, Any]:
        if problem_variable not in self.causal_graph:
            return {"error": f"Variable {problem_variable} not in causal graph"}
        all_ancestors = list(self.causal_graph_reverse.get(problem_variable, []))
        root_causes: List[Dict[str, Any]] = []
        paths_to_problem: List[Dict[str, Any]] = []

        def path_strength(path: List[str]) -> float:
            prod = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                prod *= self._edge_strength(u, v)
                if abs(prod) < min_path_strength:
                    return 0.0
            return prod

        for anc in all_ancestors:
            try:
                queue = [(anc, [anc])]
                visited = set()
                while queue:
                    node, path = queue.pop(0)
                    if len(path) - 1 > max_depth:
                        continue
                    if node == problem_variable and len(path) > 1:
                        ps = path_strength(path)
                        if abs(ps) > 0:
                            root_causes.append({
                                "root_cause": path[0],
                                "path_to_problem": path,
                                "path_string": " -> ".join(path),
                                "path_strength": float(ps),
                                "depth": len(path) - 1,
                                "is_exogenous": len(self._get_parents(path[0])) == 0,
                            })
                            paths_to_problem.append({
                                "from": path[0],
                                "to": problem_variable,
                                "path": path,
                                "strength": float(ps),
                            })
                        continue
                    for child in self._get_children(node):
                        if child not in visited:
                            visited.add(child)
                            queue.append((child, path + [child]))
            except Exception:
                continue

        root_causes.sort(key=lambda x: (-x["is_exogenous"], -abs(x["path_strength"]), x["depth"]))
        ultimate_roots = [rc for rc in root_causes if rc.get("is_exogenous")]
        return {
            "problem_variable": problem_variable,
            "all_root_causes": root_causes[:20],
            "ultimate_root_causes": ultimate_roots[:10],
            "total_paths_found": len(paths_to_problem),
            "max_depth_reached": max([rc["depth"] for rc in root_causes] + [0]),
        }

    def shapley_value_attribution(
        self,
        baseline_state: Dict[str, float],
        target_state: Dict[str, float],
        target: str,
        samples: int = 100,
    ) -> Dict[str, Any]:
        variables = list(set(list(baseline_state.keys()) + list(target_state.keys())))
        n = len(variables)
        if n == 0:
            return {"shapley_values": {}, "normalized": {}, "total_attribution": 0.0}
        rng = np.random.default_rng(self.seed)
        contributions: Dict[str, float] = {v: 0.0 for v in variables}

        def value(subset: List[str]) -> float:
            state = dict(baseline_state)
            for var in subset:
                if var in target_state:
                    state[var] = target_state[var]
            outcome = self._predict_outcomes({}, state)
            return float(outcome.get(target, 0.0))

        for _ in range(max(1, samples)):
            perm = list(variables)
            rng.shuffle(perm)
            cur_set: List[str] = []
            prev_val = value(cur_set)
            for v in perm:
                cur_set.append(v)
                new_val = value(cur_set)
                contributions[v] += new_val - prev_val
                prev_val = new_val

        shapley_values = {k: v / float(samples) for k, v in contributions.items()}
        total = sum(abs(v) for v in shapley_values.values()) or 1.0
        normalized = {k: v / total for k, v in shapley_values.items()}
        return {
            "shapley_values": shapley_values,
            "normalized": normalized,
            "total_attribution": float(sum(abs(v) for v in shapley_values.values())),
        }

    # ---- Multi-layer scenarios ----
    def multi_layer_whatif_analysis(
        self,
        scenarios: List[Dict[str, float]],
        depth: int = 3,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for scen in scenarios:
            layer1 = self._predict_outcomes({}, scen)
            affected = [k for k, v in layer1.items() if abs(v) > 0.01]
            layer2_scenarios = [{a: layer1.get(a, 0.0) * 1.2} for a in affected[:5]]
            layer2_results: List[Dict[str, Any]] = []
            for l2 in layer2_scenarios:
                l2_outcome = self._predict_outcomes(layer1, l2)
                layer2_results.append({"layer2_scenario": l2, "layer2_outcomes": l2_outcome})
            results.append({
                "scenario": scen,
                "layer1_direct_effects": layer1,
                "affected_variables": affected,
                "layer2_cascades": layer2_results,
            })
        return {"multi_layer_analysis": results, "summary": {"total_scenarios": len(results)}}

    def explore_alternate_realities(
        self,
        factual_state: Dict[str, float],
        target_outcome: str,
        target_value: Optional[float] = None,
        max_realities: int = 50,
        max_interventions: int = 3,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        variables = list(factual_state.keys())
        realities: List[Dict[str, Any]] = []
        for _ in range(max_realities):
            num_int = rng.integers(1, max(2, max_interventions + 1))
            selected = rng.choice(variables, size=min(num_int, len(variables)), replace=False)
            intervention = {}
            for var in selected:
                stats = self.standardization_stats.get(var, {"mean": factual_state.get(var, 0.0), "std": 1.0})
                intervention[var] = float(rng.normal(stats["mean"], stats["std"] * 1.5))
            outcome = self._predict_outcomes(factual_state, intervention)
            target_val = outcome.get(target_outcome, 0.0)
            if target_value is not None:
                objective = -abs(target_val - target_value)
            else:
                objective = target_val
            realities.append({
                "interventions": intervention,
                "outcome": outcome,
                "target_value": float(target_val),
                "objective": float(objective),
                "delta_from_factual": float(target_val - factual_state.get(target_outcome, 0.0)),
            })
        realities.sort(key=lambda x: x["objective"], reverse=True)
        best = realities[0] if realities else None
        return {
            "factual_state": factual_state,
            "target_outcome": target_outcome,
            "target_value": target_value,
            "best_reality": best,
            "top_10_realities": realities[:10],
            "all_realities_explored": len(realities),
            "improvement_potential": (best["target_value"] - factual_state.get(target_outcome, 0.0)) if best else 0.0,
        }

    # ---- Async wrappers ----
    async def run_async(
        self,
        task: Optional[Union[str, Any]] = None,
        initial_state: Optional[Any] = None,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(task=task, initial_state=initial_state, target_variables=target_variables, max_steps=max_steps, **kwargs))

    async def quantify_uncertainty_async(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.quantify_uncertainty(df=df, variables=variables, windows=windows, alpha=alpha))

    async def granger_causality_test_async(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.granger_causality_test(df=df, var1=var1, var2=var2, max_lag=max_lag))

    async def vector_autoregression_estimation_async(
        self,
        df: Any,
        variables: List[str],
        max_lag: int = 2,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.vector_autoregression_estimation(df=df, variables=variables, max_lag=max_lag))

    # =========================
    # Excel TUI Integration Methods
    # =========================
    
    def _initialize_excel_tables(self) -> None:
        """Initialize standard Excel tables."""
        if not self._excel_enabled or self._excel_tables is None:
            return
        
        try:
            from crca_excel.core.standard_tables import initialize_standard_tables
            initialize_standard_tables(self._excel_tables)
        except Exception as e:
            logger.error(f"Error initializing standard tables: {e}")
    
    def _link_causal_graph_to_tables(self) -> None:
        """Link CRCA causal graph to Excel tables."""
        if not self._excel_enabled or self._excel_scm_bridge is None:
            return
        
        try:
            self._excel_scm_bridge.link_causal_graph_to_tables(self.causal_graph)
        except Exception as e:
            logger.error(f"Error linking causal graph to tables: {e}")
    
    def excel_edit_cell(self, table_name: str, row_key: Any, column_name: str, value: Any) -> None:
        """
        Edit a cell in Excel tables and trigger recomputation.
        
        Args:
            table_name: Table name
            row_key: Row key
            column_name: Column name
            value: Value to set
        """
        if not self._excel_enabled or self._excel_tables is None:
            raise RuntimeError("Excel TUI not enabled")
        
        self._excel_tables.set_cell(table_name, row_key, column_name, value)
        
        # Trigger recomputation
        if self._excel_eval_engine:
            self._excel_eval_engine.recompute_dirty_cells()
    
    def excel_apply_plan(self, plan: Dict[Tuple[str, Any, str], Any]) -> Dict[str, Any]:
        """
        Apply a plan (set of interventions) to Excel tables.
        
        Args:
            plan: Dictionary of (table_name, row_key, column_name) -> value
            
        Returns:
            Dictionary with results
        """
        if not self._excel_enabled or self._excel_scm_bridge is None:
            raise RuntimeError("Excel TUI not enabled")
        
        # Convert plan format
        interventions = {
            (table_name, row_key, column_name): value
            for (table_name, row_key, column_name), value in plan.items()
        }
        
        snapshot = self._excel_scm_bridge.do_intervention(interventions)
        
        return {
            "snapshot": snapshot,
            "success": True
        }
    
    def excel_generate_scenarios(
        self,
        base_interventions: Dict[Tuple[str, Any, str], Any],
        target_variables: List[Tuple[str, Any, str]],
        n_scenarios: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios.
        
        Args:
            base_interventions: Base intervention set
            target_variables: Variables to vary
            n_scenarios: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        if not self._excel_enabled or self._excel_scm_bridge is None:
            raise RuntimeError("Excel TUI not enabled")
        
        # Convert format
        base_interv = {
            (table_name, row_key, column_name): value
            for (table_name, row_key, column_name), value in base_interventions.items()
        }
        target_vars = [
            (table_name, row_key, column_name)
            for (table_name, row_key, column_name) in target_variables
        ]
        
        return self._excel_scm_bridge.generate_counterfactual_scenarios(
            base_interv,
            target_vars,
            n_scenarios=n_scenarios,
            seed=self.seed
        )
    
    def excel_get_table(self, table_name: str):
        """
        Get an Excel table.
        
        Args:
            table_name: Table name
            
        Returns:
            Table instance or None
        """
        if not self._excel_enabled or self._excel_tables is None:
            return None
        
        return self._excel_tables.get_table(table_name)
    
    def excel_get_all_tables(self):
        """
        Get all Excel tables.
        
        Returns:
            Dictionary of table_name -> Table
        """
        if not self._excel_enabled or self._excel_tables is None:
            return {}
        
        return self._excel_tables.get_all_tables()



