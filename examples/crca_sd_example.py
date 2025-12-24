"""
CRCA-SD Example: Complete System Workflow Demonstration

This example demonstrates:
- Phase A: Initialize state, sample policies, simulate forward, check feasibility
- Phase B: Run MPC solver, extract Pareto frontier, compute CVaR risk
- Phase C: State estimation with noisy observations, board evaluation, arbitration
- Phase D: Logistics network optimization, visualization, metrics collection
- Full workflow: End-to-end example showing all phases integrated
"""

from typing import List, Dict
import numpy as np
from loguru import logger

# Import CRCA-SD components
from crca_sd.crca_sd_core import (
    StateVector,
    ControlVector,
    DynamicsModel,
    ConstraintChecker,
    ForwardSimulator,
)

from crca_sd.crca_sd_mpc import (
    ObjectiveVector,
    CVaRComputer,
    MPCSolver,
    ScenarioGenerator,
    StabilityEnforcer,
    ParetoExtractor,
)

from crca_sd.crca_sd_governance import (
    BoardMember,
    Board,
    BoardType,
    Arbitration,
    GovernanceSystem,
    LogisticsNetwork,
    Visualization,
    MetricsCollector,
)


def phase_a_basic_simulation():
    """
    Phase A: Minimal viable "toy world" demonstration.
    
    - Initialize state with 15 variables
    - Sample budget policies
    - Simulate forward 10 steps
    - Check feasibility
    """
    logger.info("=== Phase A: Basic Simulation ===")
    
    # Initialize state
    x_0 = StateVector(
        P=1000000.0,      # 1 million population
        L=500000.0,       # 500k labor force
        U=0.05,           # 5% unemployment
        W=1.0,            # Wage proxy
        S=0.8,            # 80% stability
        literacy=0.7,     # 70% literacy
        Ecap=10000.0,     # Education capacity
        Hcap=5000.0,      # Healthcare capacity
        K=1000000.0,      # Capital stock
        I=0.8,            # 80% infrastructure health
        Tcap=10000.0,     # Transport capacity
        E_stock=50000.0,  # Energy stock
        F_stock=30000.0,  # Food stock
        M_stock=20000.0,  # Materials stock
        C=0.0,            # No ecological damage yet
        Y=1000000.0,      # GDP proxy
    )
    
    logger.info(f"Initial state: P={x_0.P:,.0f}, U={x_0.U:.1%}, Y={x_0.Y:,.0f}")
    
    # Initialize dynamics and constraints
    dynamics = DynamicsModel()
    checker = ConstraintChecker()
    simulator = ForwardSimulator(dynamics, checker)
    
    # Sample 5 budget policies
    budget_categories = ["energy", "food", "infrastructure", "education", "healthcare", "R&D", "welfare"]
    policies = [
        ControlVector.sample_budget_simplex(budget_categories)
        for _ in range(5)
    ]
    
    logger.info(f"Sampled {len(policies)} policies")
    
    # Simulate each policy forward 10 steps
    results = []
    for i, policy in enumerate(policies):
        trajectory, feasibility_flags, first_violation = simulator.simulate_scenario(
            x_0, policy, [{}] * 10, horizon=10
        )
        
        feasible_count = sum(feasibility_flags)
        final_state = trajectory[-1]
        
        results.append({
            "policy_id": i,
            "feasible_steps": feasible_count,
            "first_violation": first_violation,
            "final_U": final_state.U,
            "final_Y": final_state.Y,
            "final_C": final_state.C,
        })
        
        logger.info(
            f"Policy {i}: {feasible_count}/10 feasible steps, "
            f"final U={final_state.U:.1%}, Y={final_state.Y:,.0f}"
        )
    
    # Check feasibility of initial state
    is_feasible, violations = checker.check_feasible(x_0, policies[0])
    logger.info(f"Initial state feasible: {is_feasible}")
    if violations:
        logger.warning(f"Violations: {violations[:2]}")
    
    return results


def phase_b_mpc_optimization():
    """
    Phase B: MPC optimization with risk and Pareto frontier.
    
    - Run MPC solver
    - Extract Pareto frontier
    - Compute CVaR risk
    """
    logger.info("=== Phase B: MPC Optimization ===")
    
    # Setup
    dynamics = DynamicsModel()
    checker = ConstraintChecker()
    obj_computer = ObjectiveVector(horizon=10)
    stability = StabilityEnforcer()
    solver = MPCSolver(dynamics, checker, obj_computer, horizon=10, stability_enforcer=stability)
    
    # Generate scenarios
    scenario_gen = ScenarioGenerator()
    scenarios = scenario_gen.generate_gaussian(n_scenarios=10, horizon=10)
    logger.info(f"Generated {len(scenarios)} Gaussian scenarios")
    
    # Initial state
    x_t = StateVector()
    
    # Solve MPC
    policy, solver_info = solver.solve(x_t, scenarios)
    logger.info(f"MPC solved: best_score={solver_info.get('best_score', 'N/A'):.4f}")
    
    # Simulate optimal policy
    simulator = ForwardSimulator(dynamics, checker)
    trajectory, feasibility, _ = simulator.simulate_scenario(
        x_t, policy, scenarios[0], horizon=10
    )
    
    # Compute objectives
    objectives = obj_computer.compute(trajectory, [policy] * 10)
    logger.info(f"Objectives: J_U={objectives[0]:.2f}, J_Y={objectives[2]:.2f}, J_C={objectives[4]:.2f}")
    
    # Compute CVaR
    cvar_computer = CVaRComputer(alpha=0.05)
    collapse_proxies = [
        cvar_computer.collapse_proxy(traj)
        for traj in [trajectory]  # In practice, use all scenario trajectories
    ]
    cvar = cvar_computer.compute_cvar(np.array(collapse_proxies))
    logger.info(f"CVaR (α=0.05): {cvar:.4f}")
    
    # Extract Pareto frontier (sample multiple policies)
    candidate_policies = [
        ControlVector.sample_budget_simplex(["energy", "food", "infrastructure"])
        for _ in range(10)
    ]
    
    objective_matrix = []
    for pol in candidate_policies:
        traj, _, _ = simulator.simulate_scenario(x_t, pol, scenarios[0], horizon=10)
        obj = obj_computer.compute(traj, [pol] * 10)
        objective_matrix.append(obj)
    
    objective_matrix = np.array(objective_matrix)
    pareto_policies, pareto_obj = ParetoExtractor.extract_pareto_frontier(
        candidate_policies, objective_matrix
    )
    
    logger.info(f"Pareto frontier: {len(pareto_policies)} policies")
    
    return {
        "optimal_policy": policy,
        "objectives": objectives,
        "cvar": cvar,
        "pareto_policies": pareto_policies,
    }


def phase_c_governance():
    """
    Phase C: Board governance and arbitration.
    
    - State estimation with noisy observations
    - Board evaluation
    - Arbitration
    """
    logger.info("=== Phase C: Board Governance ===")
    
    # Create boards
    growth_board = Board(
        board_id="growth_1",
        board_type=BoardType.GROWTH,
        members=[
            BoardMember(
                name="Growth Director",
                board_type=BoardType.GROWTH,
                expertise_areas=["economics", "productivity"],
                voting_weight=2.0
            )
        ]
    )
    
    welfare_board = Board(
        board_id="welfare_1",
        board_type=BoardType.WELFARE,
        members=[
            BoardMember(
                name="Welfare Director",
                board_type=BoardType.WELFARE,
                expertise_areas=["social_policy", "inequality"],
                voting_weight=2.0
            )
        ]
    )
    
    sustainability_board = Board(
        board_id="sustainability_1",
        board_type=BoardType.SUSTAINABILITY,
        members=[
            BoardMember(
                name="Sustainability Director",
                board_type=BoardType.SUSTAINABILITY,
                expertise_areas=["environment", "long_term"],
                voting_weight=2.0
            )
        ]
    )
    
    stability_board = Board(
        board_id="stability_1",
        board_type=BoardType.STABILITY,
        members=[
            BoardMember(
                name="Stability Director",
                board_type=BoardType.STABILITY,
                expertise_areas=["risk_management", "volatility"],
                voting_weight=2.0
            )
        ]
    )
    
    boards = [growth_board, welfare_board, sustainability_board, stability_board]
    logger.info(f"Created {len(boards)} governance boards")
    
    # Generate candidate policies
    dynamics = DynamicsModel()
    checker = ConstraintChecker()
    obj_computer = ObjectiveVector(horizon=5)
    simulator = ForwardSimulator(dynamics, checker)
    
    x_t = StateVector()
    candidate_policies = [
        ControlVector.sample_budget_simplex(["energy", "food", "infrastructure", "education"])
        for _ in range(5)
    ]
    
    # Evaluate policies under scenarios
    scenario_gen = ScenarioGenerator()
    scenarios = scenario_gen.generate_gaussian(n_scenarios=3, horizon=5)
    
    objective_matrix = []
    for policy in candidate_policies:
        # Aggregate objectives across scenarios
        scenario_objectives = []
        for scenario in scenarios:
            traj, _, _ = simulator.simulate_scenario(x_t, policy, scenario, horizon=5)
            obj = obj_computer.compute(traj, [policy] * 5)
            scenario_objectives.append(obj)
        expected_obj = np.mean(scenario_objectives, axis=0)
        objective_matrix.append(expected_obj)
    
    objective_matrix = np.array(objective_matrix)
    
    # Each board evaluates policies
    for board in boards:
        ranked = board.evaluate_policies(candidate_policies, objective_matrix)
        logger.info(f"{board.board_type.value} board top policy score: {ranked[0][1]:.4f}")
    
    # Arbitration: weighted vote
    governance = GovernanceSystem(boards, arbitration_method="weighted_vote")
    selected_policy = governance.select_policy(candidate_policies, objective_matrix)
    
    logger.info(f"Governance selected policy via weighted vote")
    
    # Test other arbitration methods
    consensus = Arbitration.pareto_intersection(boards, candidate_policies, objective_matrix)
    logger.info(f"Pareto intersection: {len(consensus)} consensus policies")
    
    minimax_policy = Arbitration.minimax_regret(boards, candidate_policies, objective_matrix)
    logger.info(f"Minimax regret policy selected")
    
    borda_policy = Arbitration.borda_vote(boards, candidate_policies, objective_matrix)
    logger.info(f"Borda vote policy selected")
    
    return {
        "selected_policy": selected_policy,
        "consensus_policies": consensus,
        "minimax_policy": minimax_policy,
        "borda_policy": borda_policy,
    }


def phase_d_logistics_visualization():
    """
    Phase D: Logistics network and visualization.
    
    - Logistics network optimization
    - Visualization
    - Metrics collection
    """
    logger.info("=== Phase D: Logistics & Visualization ===")
    
    # Create logistics network
    nodes = ["Region_A", "Region_B", "Port_C", "Warehouse_D"]
    edges = [
        ("Region_A", "Port_C"),
        ("Region_B", "Port_C"),
        ("Port_C", "Warehouse_D"),
    ]
    
    capacities = {
        ("Region_A", "Port_C"): 1000.0,
        ("Region_B", "Port_C"): 800.0,
        ("Port_C", "Warehouse_D"): 1500.0,
    }
    
    costs = {
        ("Region_A", "Port_C"): 1.0,
        ("Region_B", "Port_C"): 1.2,
        ("Port_C", "Warehouse_D"): 0.8,
    }
    
    network = LogisticsNetwork(nodes, edges, capacities, costs)
    logger.info(f"Created logistics network: {len(nodes)} nodes, {len(edges)} edges")
    
    # Solve flow problem
    demands = {
        "Warehouse_D": {"food": 500.0, "energy": 300.0}
    }
    commodities = ["food", "energy"]
    
    flows = network.solve_flow_problem(demands, commodities)
    logger.info(f"Solved flow problem: {len(flows)} flow variables")
    
    # Visualization
    x_t = StateVector()
    dashboard = Visualization.state_dashboard(x_t)
    logger.info("State dashboard generated")
    print(dashboard)
    
    # Metrics collection
    dynamics = DynamicsModel()
    checker = ConstraintChecker()
    simulator = ForwardSimulator(dynamics, checker)
    
    trajectories = []
    policies = []
    for _ in range(3):
        policy = ControlVector.sample_budget_simplex(["energy", "food"])
        traj, _, _ = simulator.simulate_scenario(x_t, policy, [{}] * 5, horizon=5)
        trajectories.append(traj)
        policies.append(policy)
    
    collector = MetricsCollector()
    metrics = collector.compute_all_metrics(trajectories, policies)
    logger.info(f"Computed metrics: feasibility_rate={metrics.get('feasibility_rate', 0.0):.1%}")
    
    report = collector.generate_report()
    print(report)
    
    return {
        "flows": flows,
        "metrics": metrics,
    }


def full_workflow_example():
    """
    Full end-to-end workflow example.
    
    Demonstrates complete CRCA-SD system:
    1. Initialize state and dynamics
    2. Generate scenarios
    3. Solve MPC for candidate policies
    4. Evaluate with boards
    5. Arbitrate to select final policy
    6. Simulate and collect metrics
    """
    logger.info("=== Full Workflow Example ===")
    
    # 1. Initialize
    x_0 = StateVector()
    dynamics = DynamicsModel()
    checker = ConstraintChecker()
    obj_computer = ObjectiveVector(horizon=10)
    stability = StabilityEnforcer()
    solver = MPCSolver(dynamics, checker, obj_computer, horizon=10, stability_enforcer=stability)
    simulator = ForwardSimulator(dynamics, checker)
    
    # 2. Generate scenarios
    scenario_gen = ScenarioGenerator()
    scenarios = scenario_gen.generate_gaussian(n_scenarios=5, horizon=10)
    logger.info(f"Generated {len(scenarios)} scenarios")
    
    # 3. Solve MPC
    policy_candidates, solver_info = solver.solve(x_0, scenarios)
    logger.info("MPC solved, obtained candidate policy")
    
    # Generate additional candidates for board evaluation
    budget_categories = ["energy", "food", "infrastructure", "education", "healthcare"]
    additional_candidates = [
        ControlVector.sample_budget_simplex(budget_categories)
        for _ in range(4)
    ]
    all_candidates = [policy_candidates] + additional_candidates
    
    # 4. Evaluate with boards
    boards = [
        Board(
            board_id=f"board_{i}",
            board_type=bt,
            members=[BoardMember(name=f"Member_{i}", board_type=bt, voting_weight=1.0)]
        )
        for i, bt in enumerate([BoardType.GROWTH, BoardType.WELFARE, BoardType.SUSTAINABILITY, BoardType.STABILITY])
    ]
    
    # Compute objectives for all candidates
    objective_matrix = []
    for policy in all_candidates:
        scenario_objectives = []
        for scenario in scenarios[:3]:  # Use subset for speed
            traj, _, _ = simulator.simulate_scenario(x_0, policy, scenario, horizon=10)
            obj = obj_computer.compute(traj, [policy] * 10)
            scenario_objectives.append(obj)
        expected_obj = np.mean(scenario_objectives, axis=0)
        objective_matrix.append(expected_obj)
    
    objective_matrix = np.array(objective_matrix)
    
    # 5. Arbitrate
    governance = GovernanceSystem(boards, arbitration_method="weighted_vote")
    final_policy = governance.select_policy(all_candidates, objective_matrix)
    logger.info("Governance selected final policy")
    
    # 6. Simulate final policy
    final_traj, final_feasibility, _ = simulator.simulate_scenario(
        x_0, final_policy, scenarios[0], horizon=10
    )
    
    # 7. Collect metrics
    collector = MetricsCollector()
    metrics = collector.compute_all_metrics([final_traj], [final_policy])
    
    logger.info("=== Workflow Complete ===")
    logger.info(f"Final state: U={final_traj[-1].U:.1%}, Y={final_traj[-1].Y:,.0f}, C={final_traj[-1].C:,.0f}")
    logger.info(f"Feasibility: {sum(final_feasibility)}/10 steps")
    logger.info(f"Metrics: {metrics.get('feasibility_rate', 0.0):.1%} feasibility rate")
    
    return {
        "final_policy": final_policy,
        "final_trajectory": final_traj,
        "metrics": metrics,
    }


if __name__ == "__main__":
    """Run all example phases."""
    logger.info("Starting CRCA-SD Examples")
    
    # Phase A
    phase_a_results = phase_a_basic_simulation()
    logger.info(f"Phase A complete: {len(phase_a_results)} policies evaluated")
    
    # Phase B
    phase_b_results = phase_b_mpc_optimization()
    logger.info("Phase B complete: MPC optimization done")
    
    # Phase C
    phase_c_results = phase_c_governance()
    logger.info("Phase C complete: Board governance done")
    
    # Phase D
    phase_d_results = phase_d_logistics_visualization()
    logger.info("Phase D complete: Logistics and visualization done")
    
    # Full workflow
    full_results = full_workflow_example()
    logger.info("Full workflow complete!")
    
    logger.success("All examples completed successfully!")

