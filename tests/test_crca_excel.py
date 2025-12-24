"""
Integration tests for CRCA Excel TUI System.

Tests full planning workflow, SCM bridge, and TUI interactions.
"""

import pytest
from crca_excel.core.tables import TableManager
from crca_excel.core.deps import DependencyGraph
from crca_excel.core.eval import EvaluationEngine
from crca_excel.core.scm import SCMBridge
from crca_excel.core.objective import ObjectiveFunction, ConstraintChecker
from crca_excel.core.planner import Planner
from crca_excel.core.events import EventDispatcher
from crca_excel.core.standard_tables import initialize_standard_tables
from crca_excel.core.formulas import (
    formula_agent_load,
    formula_task_progress,
    formula_objective
)


def test_table_creation():
    """Test table creation and basic operations."""
    tables = TableManager()
    initialize_standard_tables(tables)
    
    assert tables.has_table("CONFIG")
    assert tables.has_table("AGENTS")
    assert tables.has_table("TASKS")
    
    # Add an agent
    agents_table = tables.get_table("AGENTS")
    agents_table.add_row("agent1", {"agent_id": "agent1", "max_tasks": 5})
    
    assert agents_table.has_row("agent1")
    assert agents_table.get_cell("agent1", "max_tasks") == 5


def test_dependency_graph():
    """Test dependency graph operations."""
    graph = DependencyGraph()
    
    cell1 = ("TASKS", "task1", "difficulty")
    cell2 = ("AGENTS", "agent1", "load")
    
    graph.add_dependency(cell1, cell2)
    
    deps = graph.get_dependencies(cell2)
    assert cell1 in deps
    
    dependents = graph.get_dependents(cell1)
    assert cell2 in dependents


def test_scm_bridge():
    """Test SCM bridge do-operator."""
    tables = TableManager()
    initialize_standard_tables(tables)
    
    graph = DependencyGraph()
    eval_engine = EvaluationEngine(tables, graph)
    scm_bridge = SCMBridge(tables, eval_engine)
    
    # Add agent and task
    agents_table = tables.get_table("AGENTS")
    agents_table.add_row("agent1", {"agent_id": "agent1", "max_tasks": 5})
    
    tasks_table = tables.get_table("TASKS")
    tasks_table.add_row("task1", {"task_id": "task1", "difficulty": 2.0})
    
    # Apply intervention
    intervention = {
        ("TASKS", "task1", "assigned_agent_id"): "agent1"
    }
    
    snapshot = scm_bridge.do_intervention(intervention)
    assert "TASKS" in snapshot


def test_planner():
    """Test planner candidate generation and evaluation."""
    tables = TableManager()
    initialize_standard_tables(tables)
    
    graph = DependencyGraph()
    eval_engine = EvaluationEngine(tables, graph)
    scm_bridge = SCMBridge(tables, eval_engine)
    
    objective = ObjectiveFunction("value")
    constraint_checker = ConstraintChecker()
    
    planner = Planner(tables, scm_bridge, objective, constraint_checker)
    
    # Add test data
    agents_table = tables.get_table("AGENTS")
    agents_table.add_row("agent1", {"agent_id": "agent1", "max_tasks": 5})
    
    tasks_table = tables.get_table("TASKS")
    tasks_table.add_row("task1", {"task_id": "task1", "difficulty": 2.0, "value": 10.0})
    
    # Generate candidates
    intervention_vars = [("TASKS", "task1", "assigned_agent_id")]
    candidates = planner.generate_candidates(intervention_vars, n_candidates=5)
    
    assert len(candidates) > 0
    assert all(isinstance(c, dict) for c in candidates)


def test_event_dispatcher():
    """Test event dispatcher."""
    tables = TableManager()
    initialize_standard_tables(tables)
    
    graph = DependencyGraph()
    eval_engine = EvaluationEngine(tables, graph)
    dispatcher = EventDispatcher(tables, eval_engine, graph)
    
    # Add agent
    agents_table = tables.get_table("AGENTS")
    agents_table.add_row("agent1", {"agent_id": "agent1"})
    
    # Dispatch edit event
    from crca_excel.core.events import EditCellEvent
    event = EditCellEvent(
        table_name="AGENTS",
        row_key="agent1",
        column_name="max_tasks",
        value=10
    )
    
    dispatcher.dispatch(event)
    
    assert agents_table.get_cell("agent1", "max_tasks") == 10


def test_formulas():
    """Test formula evaluation."""
    tables = TableManager()
    initialize_standard_tables(tables)
    
    # Setup data
    agents_table = tables.get_table("AGENTS")
    agents_table.add_row("agent1", {"agent_id": "agent1", "max_tasks": 5})
    
    tasks_table = tables.get_table("TASKS")
    tasks_table.add_row("task1", {
        "task_id": "task1",
        "difficulty": 2.0,
        "assigned_agent_id": "agent1"
    })
    
    # Test agent load formula
    agent_row = agents_table.get_row_data("agent1")
    all_tables = {
        "AGENTS": {"agent1": agent_row},
        "TASKS": {"task1": tasks_table.get_row_data("task1")}
    }
    
    load = formula_agent_load(agent_row, all_tables, {})
    assert load == 2.0  # Sum of assigned task difficulties


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

