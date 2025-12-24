"""
Hybrid Hierarchical-Cluster Swarm implementation.

Provides intelligent task routing to appropriate swarm configurations
using an agent-based routing system with hierarchical decision-making.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import get_swarms_info
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.history_output_formatter import HistoryOutputType, history_output_formatter


_ROUTER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "select_swarm",
        "description": "Analyzes the input task and selects the most appropriate swarm configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Reasoning behind swarm selection."},
                "swarm_name": {"type": "string", "description": "Name of the selected swarm."},
                "task_description": {"type": "string", "description": "Structured task description."},
            },
            "required": ["reasoning", "swarm_name", "task_description"],
        },
    },
}]

_ROUTER_SYSTEM_PROMPT = """You are an intelligent Router Agent responsible for analyzing tasks and directing them to the most appropriate swarm.

Key Responsibilities:
1. Task Analysis: Analyze requirements, complexity, and domain
2. Swarm Selection: Match task requirements with swarm capabilities
3. Decision Making: Evaluate all available swarms systematically
4. Output Requirements: Provide clear justification and structured task description

Your output must follow this format:
{
    "swarm_name": "Name of the selected swarm",
    "task_description": "Detailed and structured task description"
}"""


class HybridHierarchicalClusterSwarm:
    """Hybrid hierarchical-cluster swarm for intelligent task routing.
    
    Routes tasks to appropriate swarm configurations using an agent-based
    routing system. Supports parallel batch processing and maintains
    conversation history for context management.
    
    Attributes:
        name: Swarm identifier name.
        description: Description of swarm functionality.
        swarms: List of available swarm routers.
        max_loops: Maximum number of processing loops.
        output_type: Format for output extraction.
        conversation: Conversation history manager.
        router_agent: Agent responsible for task routing decisions.
    """
    
    def __init__(self, name: str = "Hybrid Hierarchical-Cluster Swarm",
                 description: str = "A swarm that uses a hybrid hierarchical-peer model to solve complex tasks.",
                 swarms: List[Union[SwarmRouter, Callable]] = None,
                 agents: List[Union[Agent, Callable]] = None,  # Accept agents for convenience
                 max_loops: int = 1, output_type: HistoryOutputType = "list",
                 router_agent_model_name: str = "gpt-4o-mini", *args, **kwargs):
        """Initialize hybrid hierarchical-cluster swarm.
        
        Args:
            name: Swarm identifier name.
            description: Description of swarm functionality.
            swarms: List of available swarm routers or callables.
            max_loops: Maximum number of processing loops.
            output_type: Format for output extraction.
            router_agent_model_name: Model name for router agent.
        """
        self.name = name
        self.description = description
        
        # Handle both swarms and agents parameters
        if swarms is not None:
            self.swarms = swarms
        elif agents is not None:
            # Convert agents to swarms by wrapping them in SwarmRouter
            from utils.router import SwarmRouter
            self.swarms = [
                SwarmRouter(
                    name=f"swarm-{i}",
                    description=f"Swarm wrapper for agent {i}",
                    agents=[agent] if not isinstance(agent, list) else agent,
                    swarm_type="SequentialWorkflow",
                    max_loops=max_loops,
                    output_type=output_type,
                    **kwargs
                ) for i, agent in enumerate(agents)
            ]
        else:
            self.swarms = []
        
        self.max_loops = max_loops
        self.output_type = output_type
        self.conversation = Conversation()
        
        system_prompt = f"{_ROUTER_SYSTEM_PROMPT}\n\n{get_swarms_info(swarms=self.swarms)}"
        
        # Extract model_name from kwargs if provided, otherwise use router_agent_model_name
        model = kwargs.pop("model_name", router_agent_model_name)
        
        self.router_agent = Agent(agent_name="Router Agent",
                                 agent_description="Routes tasks to appropriate swarms.",
                                 system_prompt=system_prompt,
                                 tools_list_dictionary=_ROUTER_TOOLS,
                                 model_name=model,
                                 max_loops=1, output_type="final",
                                 **kwargs)  # Pass remaining kwargs to Agent
    
    def _parse_router_response(self, response: Union[str, dict, None]) -> dict:
        """Parse router agent response into dictionary format.
        
        Args:
            response: Response from router agent (string or dict).
            
        Returns:
            Parsed response dictionary.
            
        Raises:
            ValueError: If response cannot be parsed or is invalid.
        """
        if response is None:
            raise ValueError("Router agent returned None response. Check agent configuration and model availability.")
        
        if isinstance(response, str):
            # Handle empty or "No response generated" messages
            if not response.strip() or "No response generated" in response:
                raise ValueError(f"Router agent returned empty or invalid response: {response}")
            try:
                # Try to extract JSON from the response if it's embedded in text
                import re
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(response)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response from router agent: {response}")
        
        if isinstance(response, dict):
        return response
        
        raise ValueError(f"Unexpected response type from router agent: {type(response)}")
    
    def _validate_router_response(self, response: dict) -> tuple:
        """Validate and extract swarm name and task description.
        
        Args:
            response: Parsed response dictionary.
            
        Returns:
            Tuple of (swarm_name, task_description).
            
        Raises:
            ValueError: If required fields are missing.
        """
        swarm_name = response.get("swarm_name")
        task_description = response.get("task_description")
        
        if not swarm_name or not task_description:
            raise ValueError(f"Invalid response from router agent: both 'swarm_name' and 'task_description' must be present. "
                           f"Received: swarm_name={swarm_name}, task_description={task_description}. "
                           f"Model: {self.router_agent.model_name}")
        return swarm_name, task_description
    
    def find_swarm_by_name(self, swarm_name: str) -> Union[SwarmRouter, Callable, None]:
        """Locate swarm by name identifier.
        
        Args:
            swarm_name: Name of swarm to locate.
            
        Returns:
            Found swarm router or None if not found.
        """
        return next((swarm for swarm in self.swarms if swarm.name == swarm_name), None)
    
    def route_task(self, swarm_name: str, task_description: str) -> None:
        """Route task to specified swarm and record output.
        
        Args:
            swarm_name: Name of target swarm.
            task_description: Task description to execute.
            
        Raises:
            ValueError: If swarm is not found.
        """
        swarm = self.find_swarm_by_name(swarm_name)
        if not swarm:
            raise ValueError(f"Swarm '{swarm_name}' not found.")
        
        output = swarm.run(task_description)
        self.conversation.add(role=swarm.name, content=output)
    
    def run(self, task: str, *args, **kwargs) -> str:
        """Execute routing process for given task.
        
        Args:
            task: Task string to process.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Formatted history output string.
            
        Raises:
            ValueError: If task is empty or routing fails.
        """
        if not task:
            raise ValueError("Task cannot be empty.")
        
        self.conversation.add(role="User", content=task)
        response = self.router_agent.run(task=task)
        
        # Handle None or empty responses
        if response is None:
            raise ValueError("Router agent returned None. Check agent configuration, API keys, and model availability.")
        
        try:
        parsed_response = self._parse_router_response(response)
        except ValueError as e:
            raise ValueError(f"Failed to parse router response: {e}. Response was: {response}")
        
        try:
        swarm_name, task_description = self._validate_router_response(parsed_response)
        except ValueError as e:
            raise ValueError(f"Failed to validate router response: {e}. Parsed response was: {parsed_response}")
        
        self.route_task(swarm_name, task_description)
        
        return history_output_formatter(self.conversation, self.output_type)
    
    def batched_run(self, tasks: List[str]) -> List[str]:
        """Execute routing process for multiple tasks in parallel.
        
        Args:
            tasks: List of task strings to process.
            
        Returns:
            List of formatted history outputs.
            
        Raises:
            ValueError: If task list is empty.
        """
        if not tasks:
            raise ValueError("Task list cannot be empty.")
        
        max_workers = os.cpu_count() * 2
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self.run, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"Error processing task: {str(e)}")
        
        return results


# Alias for convenience
HHCS = HybridHierarchicalClusterSwarm
