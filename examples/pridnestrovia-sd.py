"""
Pridnestrovia (Transnistria) CRCA-SD Example

This example models Pridnestrovia's socioeconomic dynamics using real data as of December 20, 2025.
Pridnestrovia is a self-proclaimed independent region between Moldova and Ukraine, with limited
international recognition, which affects its economic and social indicators.

Real Data Sources (as of December 2025):
- Population: ~368,000 (March 2024 estimate, adjusted for decline)
- GDP: ~$1.2 billion (2021 estimate, nominal)
- GDP per capita: ~$2,584
- Currency: Transnistrian ruble (PRB), ~11 PRB = 1 USD
- Major industries: Steel, textiles, electricity generation
- Economic challenges: Limited international recognition, trade primarily with Russia/CIS
- Weather (Dec 20, 2025): Mostly cloudy, 31-37°F (0-3°C)

This example demonstrates:
1. Initializing state with real Pridnestrovia data
2. Modeling economic constraints (trade isolation, energy dependence)
3. Board governance for policy decisions
4. Scenario analysis (trade embargo, energy crisis, etc.)
5. MPC optimization under uncertainty
"""

from typing import List, Dict, Optional
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


def initialize_pridnestrovia_state() -> StateVector:
    """
    Initialize state vector with real Pridnestrovia data (December 2025).
    
    Data sources:
    - Population: 367,776 (March 2024) -> ~360,000 (Dec 2025, accounting for decline)
    - GDP: $1.201 billion (2021) -> ~$1.3 billion (2025 estimate, adjusted for inflation)
    - GDP per capita: $2,584 (2021) -> ~$3,600 (2025 estimate)
    - Unemployment: ~7% (estimated, higher than Moldova's 5.4%)
    - Literacy: ~99% (high, Soviet legacy)
    - Infrastructure: Moderate (aging Soviet-era infrastructure)
    
    Returns:
        StateVector: Initialized state vector
    """
    # Population: declining due to emigration
    population = 360000.0  # ~360k (down from 367k in 2024)
    
    # Labor force: ~50% participation rate (typical for post-Soviet region)
    labor_force = population * 0.50
    
    # Unemployment: higher due to economic isolation
    unemployment_rate = 0.07  # 7% (estimated, higher than Moldova)
    
    # Wage: low due to limited economic opportunities
    # Average monthly wage ~$300-400 USD equivalent
    wage_proxy = 0.35  # Normalized proxy (low)
    
    # Social stability: moderate (unrecognized status creates uncertainty)
    stability = 0.65  # 65% (moderate, affected by political status)
    
    # Literacy: very high (Soviet education legacy)
    literacy = 0.99  # 99% (very high)
    
    # Education capacity: moderate (Soviet-era institutions)
    education_capacity = 15000.0  # Seats/teachers
    
    # Healthcare capacity: limited (aging infrastructure)
    healthcare_capacity = 8000.0  # Beds/staff
    
    # Capital stock: industrial (steel, textiles, electricity)
    # Estimated at ~$800M in productive capital
    capital_stock = 800000000.0  # $800M
    
    # Infrastructure: aging Soviet-era, moderate condition
    infrastructure_health = 0.65  # 65% (moderate, needs maintenance)
    
    # Transport capacity: limited (landlocked, trade isolation)
    transport_capacity = 5000.0  # ton-km/day (limited)
    
    # Energy stock: dependent on imports (Russia, Ukraine)
    energy_stock = 20000.0  # Limited reserves
    
    # Food stock: moderate (agricultural region)
    food_stock = 15000.0  # Moderate reserves
    
    # Materials stock: industrial inputs (steel production)
    materials_stock = 10000.0  # Industrial materials
    
    # Ecological damage: moderate (industrial pollution)
    ecological_damage = 50000.0  # Accumulated damage
    
    # GDP proxy: ~$1.3B (2025 estimate)
    gdp_proxy = 1300000000.0  # $1.3B
    
    state = StateVector(
        P=population,
        L=labor_force,
        U=unemployment_rate,
        W=wage_proxy,
        S=stability,
        literacy=literacy,
        Ecap=education_capacity,
        Hcap=healthcare_capacity,
        K=capital_stock,
        I=infrastructure_health,
        Tcap=transport_capacity,
        E_stock=energy_stock,
        F_stock=food_stock,
        M_stock=materials_stock,
        C=ecological_damage,
        Y=gdp_proxy,
    )
    
    logger.info("Initialized Pridnestrovia state:")
    logger.info(f"  Population: {population:,.0f}")
    logger.info(f"  GDP: ${gdp_proxy/1e9:.2f}B")
    logger.info(f"  Unemployment: {unemployment_rate:.1%}")
    logger.info(f"  Stability: {stability:.1%}")
    
    return state


def create_pridnestrovia_dynamics() -> DynamicsModel:
    """
    Create dynamics model with Pridnestrovia-specific parameters.
    
    Adjustments for Pridnestrovia:
    - Higher capital depreciation (aging industrial base)
    - Lower investment efficiency (limited access to international markets)
    - Higher infrastructure decay (aging Soviet-era infrastructure)
    - Lower trade efficiency (isolation)
    
    Returns:
        DynamicsModel: Configured dynamics model
    """
    return DynamicsModel(
        delta_K=0.06,          # 6% capital depreciation (aging industrial base)
        delta_I=0.03,          # 3% infrastructure decay (faster than typical)
        alpha=0.35,            # 35% capital share (industrial economy)
        kappa_K=0.65,          # 65% investment efficiency (lower due to isolation)
        kappa_I=0.60,          # 60% maintenance efficiency (aging infrastructure)
        kappa_literacy=0.08,   # 8% literacy diffusion (already high)
        delta_literacy=0.005,  # 0.5% literacy disruption (low, good education)
        alpha_U=0.12,          # 12% unemployment adjustment (slower, limited opportunities)
        alpha_rho=0.04,        # 4% labor participation adjustment
    )


def create_pridnestrovia_scenarios() -> List[List[Dict[str, float]]]:
    """
    Generate realistic scenarios for Pridnestrovia.
    
    Scenarios reflect real risks:
    1. Trade embargo (Russia/Ukraine tensions)
    2. Energy crisis (gas supply disruption)
    3. Economic isolation (sanctions)
    4. Normal operation (baseline)
    
    Returns:
        List[List[Dict[str, float]]]: List of scenario sequences
    """
    generator = ScenarioGenerator()
    scenarios = []
    
    # Scenario 1: Baseline (normal operation)
    baseline = generator.generate_gaussian(
        n_scenarios=1,
        horizon=12,
        mean={
            "demand_shock": 0.0,
            "trade_shock": 0.0,
            "productivity_shock": 1.0,
            "disaster_shock": 0.0,
            "labor_shock": 0.0,
            "unemployment_shock": 0.0,
            "energy_import": 500.0,  # Normal energy imports
            "food_import": 200.0,    # Normal food imports
        },
        cov=np.eye(8) * 0.01  # Low variance
    )
    scenarios.extend(baseline)
    
    # Scenario 2: Trade embargo (reduced trade with Russia/Ukraine)
    embargo = generator.generate_structured_shock(
        "trade_embargo",
        magnitude=0.8,  # 80% trade reduction
        timing=3,       # Starts at month 3
        horizon=12
    )
    scenarios.append(embargo)
    
    # Scenario 3: Energy crisis (gas supply disruption)
    energy_crisis = []
    for t in range(12):
        if t >= 2:  # Crisis starts at month 2
            energy_crisis.append({
                "energy_import": -1000.0,  # Severe reduction
                "trade_shock": 0.3,        # Moderate trade impact
                "productivity_shock": 0.85, # Reduced productivity
            })
        else:
            energy_crisis.append({})
    scenarios.append(energy_crisis)
    
    # Scenario 4: Economic sanctions (increased isolation)
    sanctions = []
    for t in range(12):
        if t >= 4:  # Sanctions start at month 4
            sanctions.append({
                "trade_shock": 0.6,        # 60% trade reduction
                "demand_shock": -0.2,      # Reduced demand
                "productivity_shock": 0.9,  # 10% productivity loss
            })
        else:
            sanctions.append({})
    scenarios.append(sanctions)
    
    # Scenario 5: Student-t heavy-tailed disaster
    heavy_tail = generator.generate_student_t(
        n_scenarios=1,
        horizon=12,
        df=3.0,   # Heavy tails
        scale=0.15
    )
    scenarios.extend(heavy_tail)
    
    logger.info(f"Generated {len(scenarios)} scenarios for Pridnestrovia")
    return scenarios


def create_pridnestrovia_boards() -> List[Board]:
    """
    Create governance boards for Pridnestrovia.
    
    Boards reflect real policy priorities:
    - Growth: Industrial output, employment
    - Welfare: Social stability, unemployment reduction
    - Sustainability: Long-term resilience, environmental protection
    - Stability: Risk management, avoiding collapse
    
    Returns:
        List[Board]: List of governance boards
    """
    growth_board = Board(
        board_id="growth_pridnestrovia",
        board_type=BoardType.GROWTH,
        members=[
            BoardMember(
                name="Industrial Director",
                board_type=BoardType.GROWTH,
                expertise_areas=["steel_production", "manufacturing", "trade"],
                voting_weight=2.0,
                independence_status=False,
            ),
            BoardMember(
                name="Economic Advisor",
                board_type=BoardType.GROWTH,
                expertise_areas=["macroeconomics", "investment"],
                voting_weight=1.5,
                independence_status=True,
            ),
        ]
    )
    
    welfare_board = Board(
        board_id="welfare_pridnestrovia",
        board_type=BoardType.WELFARE,
        members=[
            BoardMember(
                name="Social Policy Director",
                board_type=BoardType.WELFARE,
                expertise_areas=["social_policy", "unemployment", "education"],
                voting_weight=2.0,
                independence_status=False,
            ),
            BoardMember(
                name="Healthcare Director",
                board_type=BoardType.WELFARE,
                expertise_areas=["healthcare", "public_health"],
                voting_weight=1.5,
                independence_status=False,
            ),
        ]
    )
    
    sustainability_board = Board(
        board_id="sustainability_pridnestrovia",
        board_type=BoardType.SUSTAINABILITY,
        members=[
            BoardMember(
                name="Environmental Director",
                board_type=BoardType.SUSTAINABILITY,
                expertise_areas=["environment", "pollution_control", "energy_efficiency"],
                voting_weight=2.0,
                independence_status=True,
            ),
        ]
    )
    
    stability_board = Board(
        board_id="stability_pridnestrovia",
        board_type=BoardType.STABILITY,
        members=[
            BoardMember(
                name="Risk Management Director",
                board_type=BoardType.STABILITY,
                expertise_areas=["risk_management", "crisis_management", "resilience"],
                voting_weight=2.5,
                independence_status=True,
            ),
        ]
    )
    
    boards = [growth_board, welfare_board, sustainability_board, stability_board]
    logger.info(f"Created {len(boards)} governance boards for Pridnestrovia")
    return boards


def pridnestrovia_mpc_analysis():
    """
    Run MPC analysis for Pridnestrovia.
    
    Analyzes policy options under uncertainty, considering:
    - Trade isolation risks
    - Energy dependence
    - Economic constraints
    - Social stability requirements
    """
    logger.info("=== Pridnestrovia MPC Analysis ===")
    
    # Initialize
    x_0 = initialize_pridnestrovia_state()
    dynamics = create_pridnestrovia_dynamics()
    checker = ConstraintChecker(U_max=0.15, S_min=0.50)  # Stricter bounds for stability
    obj_computer = ObjectiveVector(horizon=12)
    stability = StabilityEnforcer(max_budget_change=0.15)  # Conservative rate limits
    solver = MPCSolver(dynamics, checker, obj_computer, horizon=12, stability_enforcer=stability)
    simulator = ForwardSimulator(dynamics, checker)
    
    # Generate scenarios
    scenarios = create_pridnestrovia_scenarios()
    
    # Solve MPC
    logger.info("Solving MPC optimization...")
    policy, solver_info = solver.solve(x_0, scenarios)
    
    logger.info(f"MPC solved: best_score={solver_info.get('best_score', 'N/A')}")
    logger.info(f"Budget allocation: {policy.budget_shares}")
    
    # Simulate optimal policy
    logger.info("Simulating optimal policy...")
    trajectory, feasibility, first_violation = simulator.simulate_scenario(
        x_0, policy, scenarios[0], horizon=12
    )
    
    feasible_steps = sum(feasibility)
    logger.info(f"Feasibility: {feasible_steps}/12 steps")
    
    if first_violation:
        logger.warning(f"First constraint violation at month {first_violation}")
    
    # Compute objectives
    objectives = obj_computer.compute(trajectory, [policy] * 12)
    logger.info(f"Objectives: J_U={objectives[0]:.2f}, J_Y={objectives[2]:.2f}, J_C={objectives[4]:.2f}")
    
    # Compute CVaR
    cvar_computer = CVaRComputer(alpha=0.05)
    collapse_proxies = []
    for scenario in scenarios[:3]:  # Use subset for speed
        traj, _, _ = simulator.simulate_scenario(x_0, policy, scenario, horizon=12)
        proxy = cvar_computer.collapse_proxy(traj)
        collapse_proxies.append(proxy)
    
    cvar = cvar_computer.compute_cvar(np.array(collapse_proxies))
    logger.info(f"CVaR (α=0.05): {cvar:.4f}")
    
    # Final state
    final_state = trajectory[-1]
    logger.info(f"Final state: P={final_state.P:,.0f}, U={final_state.U:.1%}, Y=${final_state.Y/1e9:.2f}B")
    
    return {
        "policy": policy,
        "trajectory": trajectory,
        "objectives": objectives,
        "cvar": cvar,
        "feasibility": feasible_steps,
    }


def pridnestrovia_governance_demo():
    """
    Demonstrate board governance for Pridnestrovia policy selection.
    
    Shows how different boards evaluate policies and how arbitration
    selects the final policy.
    """
    logger.info("=== Pridnestrovia Board Governance Demo ===")
    
    # Initialize
    x_0 = initialize_pridnestrovia_state()
    dynamics = create_pridnestrovia_dynamics()
    checker = ConstraintChecker()
    obj_computer = ObjectiveVector(horizon=12)
    simulator = ForwardSimulator(dynamics, checker)
    
    # Create boards
    boards = create_pridnestrovia_boards()
    
    # Generate candidate policies
    budget_categories = [
        "energy",      # Energy security (critical for Pridnestrovia)
        "food",        # Food security
        "infrastructure",  # Infrastructure maintenance
        "education",   # Education (already high, maintain)
        "healthcare",  # Healthcare (aging population)
        "R&D",         # Research & development (limited)
        "welfare",     # Social welfare
    ]
    
    candidate_policies = [
        ControlVector.sample_budget_simplex(budget_categories)
        for _ in range(6)
    ]
    
    # Evaluate under scenarios
    scenarios = create_pridnestrovia_scenarios()
    objective_matrix = []
    
    logger.info("Evaluating candidate policies...")
    for i, policy in enumerate(candidate_policies):
        scenario_objectives = []
        for scenario in scenarios[:3]:  # Use subset for speed
            traj, _, _ = simulator.simulate_scenario(x_0, policy, scenario, horizon=12)
            obj = obj_computer.compute(traj, [policy] * 12)
            scenario_objectives.append(obj)
        expected_obj = np.mean(scenario_objectives, axis=0)
        objective_matrix.append(expected_obj)
        logger.info(f"Policy {i+1} objectives: J_U={expected_obj[0]:.2f}, J_Y={expected_obj[2]:.2f}")
    
    objective_matrix = np.array(objective_matrix)
    
    # Board evaluations
    logger.info("\nBoard evaluations:")
    for board in boards:
        ranked = board.evaluate_policies(candidate_policies, objective_matrix)
        top_policy_idx = candidate_policies.index(ranked[0][0])
        logger.info(f"{board.board_type.value} board: Top policy {top_policy_idx+1}, score={ranked[0][1]:.4f}")
    
    # Arbitration
    governance = GovernanceSystem(boards, arbitration_method="weighted_vote")
    selected_policy = governance.select_policy(candidate_policies, objective_matrix)
    selected_idx = candidate_policies.index(selected_policy)
    
    logger.info(f"\nGovernance selected: Policy {selected_idx+1} via weighted vote")
    
    # Test other methods
    consensus = Arbitration.pareto_intersection(boards, candidate_policies, objective_matrix)
    logger.info(f"Pareto intersection: {len(consensus)} consensus policies")
    
    minimax_policy = Arbitration.minimax_regret(boards, candidate_policies, objective_matrix)
    logger.info(f"Minimax regret: Policy {candidate_policies.index(minimax_policy)+1}")
    
    return {
        "selected_policy": selected_policy,
        "consensus_policies": consensus,
        "minimax_policy": minimax_policy,
    }


def pridnestrovia_logistics_demo():
    """
    Demonstrate logistics network for Pridnestrovia.
    
    Models trade flows between:
    - Tiraspol (capital)
    - Bender (industrial city)
    - Rybnitsa (steel production)
    - External: Russia, Ukraine, Moldova
    """
    logger.info("=== Pridnestrovia Logistics Network Demo ===")
    
    # Create network
    nodes = [
        "Tiraspol",      # Capital
        "Bender",        # Industrial city
        "Rybnitsa",      # Steel production
        "Russia",        # External: Russia
        "Ukraine",       # External: Ukraine
        "Moldova",       # External: Moldova
    ]
    
    edges = [
        ("Tiraspol", "Bender"),
        ("Tiraspol", "Rybnitsa"),
        ("Bender", "Tiraspol"),
        ("Rybnitsa", "Tiraspol"),
        ("Russia", "Tiraspol"),    # Energy imports
        ("Ukraine", "Tiraspol"),   # Trade
        ("Moldova", "Tiraspol"),   # Limited trade
        ("Tiraspol", "Russia"),    # Exports
        ("Tiraspol", "Ukraine"),   # Exports
    ]
    
    # Capacities (limited due to isolation)
    capacities = {
        ("Russia", "Tiraspol"): 2000.0,   # Energy imports
        ("Ukraine", "Tiraspol"): 1500.0,  # Trade
        ("Moldova", "Tiraspol"): 500.0,   # Limited trade
        ("Tiraspol", "Bender"): 1000.0,
        ("Tiraspol", "Rybnitsa"): 800.0,
    }
    
    # Costs (higher due to isolation)
    costs = {
        ("Russia", "Tiraspol"): 1.2,      # Higher cost
        ("Ukraine", "Tiraspol"): 1.5,     # Even higher (tensions)
        ("Moldova", "Tiraspol"): 2.0,     # Very high (limited trade)
        ("Tiraspol", "Bender"): 0.8,
        ("Tiraspol", "Rybnitsa"): 0.9,
    }
    
    network = LogisticsNetwork(nodes, edges, capacities, costs)
    logger.info(f"Created logistics network: {len(nodes)} nodes, {len(edges)} edges")
    
    # Solve flow problem
    demands = {
        "Tiraspol": {"energy": 800.0, "food": 300.0, "materials": 200.0},
        "Bender": {"energy": 400.0, "food": 150.0},
        "Rybnitsa": {"energy": 500.0, "materials": 300.0},
    }
    commodities = ["energy", "food", "materials"]
    
    flows = network.solve_flow_problem(demands, commodities)
    logger.info(f"Solved flow problem: {len(flows)} flow variables")
    
    # Analyze flows
    total_energy_import = sum(
        v for (src, tgt, comm), v in flows.items()
        if comm == "energy" and src in ["Russia", "Ukraine", "Moldova"]
    )
    logger.info(f"Total energy imports: {total_energy_import:.0f} units")
    
    return {
        "network": network,
        "flows": flows,
        "total_energy_import": total_energy_import,
    }


def full_pridnestrovia_workflow():
    """
    Full end-to-end workflow for Pridnestrovia.
    
    Demonstrates complete CRCA-SD system:
    1. Initialize with real data
    2. Generate scenarios (trade embargo, energy crisis, etc.)
    3. Solve MPC for candidate policies
    4. Evaluate with boards
    5. Arbitrate to select final policy
    6. Simulate and collect metrics
    """
    logger.info("=== Full Pridnestrovia Workflow ===")
    
    # 1. Initialize
    x_0 = initialize_pridnestrovia_state()
    dynamics = create_pridnestrovia_dynamics()
    checker = ConstraintChecker(U_max=0.15, S_min=0.50)
    obj_computer = ObjectiveVector(horizon=12)
    stability = StabilityEnforcer(max_budget_change=0.15)
    solver = MPCSolver(dynamics, checker, obj_computer, horizon=12, stability_enforcer=stability)
    simulator = ForwardSimulator(dynamics, checker)
    
    # 2. Generate scenarios
    scenarios = create_pridnestrovia_scenarios()
    logger.info(f"Generated {len(scenarios)} scenarios")
    
    # 3. Solve MPC
    logger.info("Solving MPC...")
    policy_candidates, solver_info = solver.solve(x_0, scenarios)
    
    # Generate additional candidates
    budget_categories = ["energy", "food", "infrastructure", "education", "healthcare", "R&D", "welfare"]
    additional_candidates = [
        ControlVector.sample_budget_simplex(budget_categories)
        for _ in range(4)
    ]
    all_candidates = [policy_candidates] + additional_candidates
    
    # 4. Evaluate with boards
    boards = create_pridnestrovia_boards()
    
    objective_matrix = []
    for policy in all_candidates:
        scenario_objectives = []
        for scenario in scenarios[:3]:
            traj, _, _ = simulator.simulate_scenario(x_0, policy, scenario, horizon=12)
            obj = obj_computer.compute(traj, [policy] * 12)
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
        x_0, final_policy, scenarios[0], horizon=12
    )
    
    # 7. Collect metrics
    collector = MetricsCollector()
    metrics = collector.compute_all_metrics([final_traj], [final_policy])
    
    # 8. Visualization
    dashboard = Visualization.state_dashboard(x_0)
    print("\n=== Initial State ===")
    print(dashboard)
    
    final_dashboard = Visualization.state_dashboard(final_traj[-1])
    print("\n=== Final State (12 months) ===")
    print(final_dashboard)
    
    report = collector.generate_report()
    print("\n=== Metrics Report ===")
    print(report)
    
    logger.info("=== Workflow Complete ===")
    logger.info(f"Final state: U={final_traj[-1].U:.1%}, Y=${final_traj[-1].Y/1e9:.2f}B, C={final_traj[-1].C:,.0f}")
    logger.info(f"Feasibility: {sum(final_feasibility)}/12 steps")
    logger.info(f"Metrics: {metrics.get('feasibility_rate', 0.0):.1%} feasibility rate")
    
    return {
        "final_policy": final_policy,
        "final_trajectory": final_traj,
        "metrics": metrics,
    }


if __name__ == "__main__":
    """Run Pridnestrovia examples."""
    logger.info("Starting Pridnestrovia CRCA-SD Examples")
    logger.info("Date: December 20, 2025")
    logger.info("=" * 60)
    
    # Run individual demos
    try:
        mpc_results = pridnestrovia_mpc_analysis()
        logger.info("MPC analysis complete\n")
    except Exception as e:
        logger.error(f"MPC analysis failed: {e}")
    
    try:
        governance_results = pridnestrovia_governance_demo()
        logger.info("Governance demo complete\n")
    except Exception as e:
        logger.error(f"Governance demo failed: {e}")
    
    try:
        logistics_results = pridnestrovia_logistics_demo()
        logger.info("Logistics demo complete\n")
    except Exception as e:
        logger.error(f"Logistics demo failed: {e}")
    
    # Run full workflow
    try:
        full_results = full_pridnestrovia_workflow()
        logger.info("Full workflow complete!")
    except Exception as e:
        logger.error(f"Full workflow failed: {e}")
    
    logger.success("All Pridnestrovia examples completed!")

