"""Agent type definitions for CR-CA.

This module provides type aliases for agent-related types used throughout
the CR-CA codebase.
"""

from typing import Any, Callable, Sequence, Union

from swarms.structs.agent import Agent

# Unified type for agent
AgentType = Union[Agent, Callable, Any]

# List of agents
AgentListType = Sequence[AgentType]