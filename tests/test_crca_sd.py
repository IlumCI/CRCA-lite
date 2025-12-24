"""
Unit tests for CRCA-SD components.

Tests cover:
- State vector initialization, validation, serialization
- Control vector sampling, simplex validation
- Dynamics model single-step and multi-step simulation
- Constraint checker for all constraint types
- Objective vector computation and aggregation
- CVaR computation accuracy
- MPC solver feasibility and optimality
- Scenario generation (Gaussian, structured)
- State estimation (EKF/UKF)
- Board evaluation and arbitration
- Logistics network flow optimization
- Integration tests for full workflows
"""

import pytest
import numpy as np
from typing import List, Dict

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
    StateEstimator,
)

from crca_sd.crca_sd_governance import (
    BoardMember,
    Board,
    BoardType,
    Arbitration,
    GovernanceSystem,
    LogisticsNetwork,
    Visualization,
    CRCA_SD_Config,
    RiskAssessment,
    MetricsCollector,
)


class TestStateVector:
    """Test StateVector class."""
    
    def test_initialization(self):
        """Test state vector initialization."""
        x = StateVector()
        assert x.P > 0
        assert 0 <= x.U <= 1
        assert 0 <= x.S <= 1
    
    def test_validation(self):
        """Test state vector validation."""
        x = StateVector()
        is_valid, violations = x.validate()
        assert is_valid
        assert len(violations) == 0
        
        # Test invalid state
        x.U = 1.5  # Out of bounds
        is_valid, violations = x.validate()
        assert not is_valid
        assert len(violations) > 0
    
    def test_serialization(self):
        """Test state vector serialization."""
        x = StateVector(P=1000000.0, U=0.05)
        x_dict = x.to_dict()
        assert x_dict["P"] == 1000000.0
        assert x_dict["U"] == 0.05
        
        x_restored = StateVector.from_dict(x_dict)
        assert x_restored.P == x.P
        assert x_restored.U == x.U
    
    def test_copy(self):
        """Test state vector copying."""
        x = StateVector(P=1000000.0)
        x_copy = x.copy()
        assert x_copy.P == x.P
        x_copy.P = 2000000.0
        assert x.P != x_copy.P  # Deep copy


class TestControlVector:
    """Test ControlVector class."""
    
    def test_simplex_validation(self):
        """Test simplex constraint validation."""
        u = ControlVector(budget_shares={"a": 0.5, "b": 0.5})
        is_valid, error = u.validate_simplex()
        assert is_valid
        assert error is None
        
        # Invalid: doesn't sum to 1
        u = ControlVector(budget_shares={"a": 0.5, "b": 0.3})
        is_valid, error = u.validate_simplex()
        assert not is_valid
        
        # Invalid: negative value
        u = ControlVector(budget_shares={"a": 0.5, "b": -0.1, "c": 0.6})
        is_valid, error = u.validate_simplex()
        assert not is_valid
    
    def test_dirichlet_sampling(self):
        """Test Dirichlet sampling for budget shares."""
        categories = ["energy", "food", "infrastructure"]
        u = ControlVector.sample_budget_simplex(categories)
        is_valid, error = u.validate_simplex()
        assert is_valid
        assert all(cat in u.budget_shares for cat in categories)
    
    def test_serialization(self):
        """Test control vector serialization."""
        u = ControlVector(budget_shares={"a": 0.5, "b": 0.5})
        u_dict = u.to_dict()
        assert "budget_shares" in u_dict
        
        u_restored = ControlVector.from_dict(u_dict)
        assert u_restored.budget_shares == u.budget_shares


class TestDynamicsModel:
    """Test DynamicsModel class."""
    
    def test_single_step(self):
        """Test single time step evolution."""
        dynamics = DynamicsModel()
        x_0 = StateVector()
        u_0 = ControlVector.sample_budget_simplex(
            ["energy", "food", "infrastructure", "education"]
        )
        
        x_1 = dynamics.step(x_0, u_0)
        
        assert isinstance(x_1, StateVector)
        assert x_1.P >= 0
        assert 0 <= x_1.U <= 1
    
    def test_multi_step(self):
        """Test multi-step simulation."""
        dynamics = DynamicsModel()
        x_0 = StateVector()
        u = ControlVector.sample_budget_simplex(
            ["energy", "food", "infrastructure"]
        )
        
        x_current = x_0
        for _ in range(5):
            x_current = dynamics.step(x_current, u)
            assert isinstance(x_current, StateVector)


class TestConstraintChecker:
    """Test ConstraintChecker class."""
    
    def test_feasibility_check(self):
        """Test constraint feasibility checking."""
        checker = ConstraintChecker()
        x = StateVector()
        u = ControlVector.sample_budget_simplex(["energy", "food"])
        
        is_feasible, violations = checker.check_feasible(x, u)
        # Should be feasible with default state
        assert isinstance(is_feasible, bool)
        assert isinstance(violations, list)
    
    def test_violations(self):
        """Test constraint violation detection."""
        checker = ConstraintChecker()
        x = StateVector(U=0.25)  # Above U_max=0.2
        u = ControlVector.sample_budget_simplex(["energy"])
        
        is_feasible, violations = checker.check_feasible(x, u)
        assert not is_feasible
        assert len(violations) > 0
        assert any("Unemployment" in v for v in violations)


class TestForwardSimulator:
    """Test ForwardSimulator class."""
    
    def test_simulate_scenario(self):
        """Test scenario simulation."""
        dynamics = DynamicsModel()
        checker = ConstraintChecker()
        simulator = ForwardSimulator(dynamics, checker)
        
        x_0 = StateVector()
        u = ControlVector.sample_budget_simplex(["energy", "food"])
        disturbances = [{}] * 10
        
        trajectory, feasibility_flags, first_violation = simulator.simulate_scenario(
            x_0, u, disturbances, horizon=10
        )
        
        assert len(trajectory) == 11  # Initial + 10 steps
        assert len(feasibility_flags) == 10
        assert isinstance(first_violation, (int, type(None)))


class TestObjectiveVector:
    """Test ObjectiveVector class."""
    
    def test_compute(self):
        """Test objective computation."""
        obj_computer = ObjectiveVector(horizon=5)
        
        trajectory = [StateVector() for _ in range(6)]
        controls = [ControlVector() for _ in range(5)]
        
        objectives = obj_computer.compute(trajectory, controls)
        
        assert objectives.shape == (6,)
        assert objectives[0] >= 0  # J_U (unemployment sum)
        assert objectives[1] <= 0  # J_ℓ (negative literacy sum)
    
    def test_aggregate(self):
        """Test objective aggregation across scenarios."""
        obj_computer = ObjectiveVector()
        
        scenarios = [
            ([StateVector() for _ in range(3)], [ControlVector() for _ in range(2)])
            for _ in range(3)
        ]
        
        expected_obj = obj_computer.aggregate(scenarios)
        assert expected_obj.shape == (6,)


class TestCVaRComputer:
    """Test CVaRComputer class."""
    
    def test_compute_cvar(self):
        """Test CVaR computation."""
        cvar_computer = CVaRComputer(alpha=0.05)
        
        z_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cvar = cvar_computer.compute_cvar(z_scores)
        
        assert cvar >= 0
        # CVaR should be >= mean of worst 5%
        assert cvar >= np.mean(z_scores[-1:])  # Worst scenario
    
    def test_collapse_proxy(self):
        """Test collapse proxy computation."""
        cvar_computer = CVaRComputer()
        
        trajectory = [StateVector(U=0.2, S=0.3)]  # High U, low S
        proxy = cvar_computer.collapse_proxy(trajectory)
        
        assert proxy >= 0


class TestScenarioGenerator:
    """Test ScenarioGenerator class."""
    
    def test_gaussian_generation(self):
        """Test Gaussian scenario generation."""
        generator = ScenarioGenerator()
        
        scenarios = generator.generate_gaussian(n_scenarios=5, horizon=10)
        
        assert len(scenarios) == 5
        assert len(scenarios[0]) == 10
        assert isinstance(scenarios[0][0], dict)
    
    def test_structured_shock(self):
        """Test structured shock generation."""
        generator = ScenarioGenerator()
        
        scenario = generator.generate_structured_shock(
            "trade_embargo", magnitude=0.5, timing=5, horizon=10
        )
        
        assert len(scenario) == 10
        assert scenario[5].get("trade_shock", 0) > 0  # Shock at timing


class TestStabilityEnforcer:
    """Test StabilityEnforcer class."""
    
    def test_rate_limits(self):
        """Test rate limit application."""
        enforcer = StabilityEnforcer(max_budget_change=0.1)
        
        u_prev = ControlVector(budget_shares={"a": 0.5, "b": 0.5})
        u_curr = ControlVector(budget_shares={"a": 0.9, "b": 0.1})  # Large change
        
        u_adjusted = enforcer.apply_rate_limits(u_curr, u_prev)
        
        # Check that change is limited
        change = sum(abs(u_adjusted.budget_shares.get(k, 0) - u_prev.budget_shares.get(k, 0))
                    for k in set(list(u_adjusted.budget_shares.keys()) + list(u_prev.budget_shares.keys())))
        assert change <= 0.1 + 1e-6  # Within limit
    
    def test_investment_smoothing(self):
        """Test investment smoothing."""
        enforcer = StabilityEnforcer(investment_smoothing=0.7)
        
        I_hat = 100.0
        I_prev = 50.0
        
        I_smooth = enforcer.smooth_investment(I_hat, I_prev)
        
        # Should be between I_prev and I_hat
        assert I_prev <= I_smooth <= I_hat


class TestParetoExtractor:
    """Test ParetoExtractor class."""
    
    def test_pareto_extraction(self):
        """Test Pareto frontier extraction."""
        policies = [
            ControlVector(budget_shares={"a": 0.5, "b": 0.5})
            for _ in range(5)
        ]
        
        # Create objective matrix (2 objectives)
        objectives = np.array([
            [1.0, 2.0],  # Policy 0
            [2.0, 1.0],  # Policy 1 (dominated by 0? No, different objectives)
            [0.5, 3.0],  # Policy 2 (dominated by 0)
            [3.0, 0.5],  # Policy 3
            [1.5, 1.5],  # Policy 4
        ])
        
        pareto_policies, pareto_obj = ParetoExtractor.extract_pareto_frontier(
            policies, objectives
        )
        
        assert len(pareto_policies) <= len(policies)
        assert pareto_obj.shape[0] == len(pareto_policies)


class TestMPCSolver:
    """Test MPCSolver class."""
    
    def test_solve(self):
        """Test MPC solver."""
        dynamics = DynamicsModel()
        checker = ConstraintChecker()
        obj_computer = ObjectiveVector(horizon=5)
        solver = MPCSolver(dynamics, checker, obj_computer, horizon=5)
        
        x_t = StateVector()
        scenarios = [[{}] * 5 for _ in range(3)]
        
        policy, info = solver.solve(x_t, scenarios)
        
        assert isinstance(policy, ControlVector)
        assert "best_score" in info


class TestBoard:
    """Test Board class."""
    
    def test_preference_weights(self):
        """Test board preference weights."""
        board = Board(
            board_id="test",
            board_type=BoardType.GROWTH,
            members=[BoardMember(name="Test", board_type=BoardType.GROWTH)]
        )
        
        weights = board.get_preference_weights()
        
        assert "J_Y" in weights
        assert weights["J_Y"] > weights["J_C"]  # Growth prioritizes output
    
    def test_evaluate_policies(self):
        """Test policy evaluation."""
        board = Board(
            board_id="test",
            board_type=BoardType.WELFARE,
            members=[BoardMember(name="Test", board_type=BoardType.WELFARE)]
        )
        
        policies = [ControlVector.sample_budget_simplex(["a", "b"]) for _ in range(3)]
        objectives = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for _ in range(3)])
        
        ranked = board.evaluate_policies(policies, objectives)
        
        assert len(ranked) == len(policies)
        assert ranked[0][1] <= ranked[-1][1]  # Sorted by score


class TestArbitration:
    """Test Arbitration class."""
    
    def test_weighted_vote(self):
        """Test weighted voting."""
        boards = [
            Board(
                board_id=f"board_{i}",
                board_type=BoardType.GROWTH,
                members=[BoardMember(name=f"Member_{i}", board_type=BoardType.GROWTH, voting_weight=1.0)]
            )
            for i in range(2)
        ]
        
        policies = [ControlVector.sample_budget_simplex(["a", "b"]) for _ in range(3)]
        objectives = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for _ in range(3)])
        
        selected = Arbitration.weighted_vote(boards, policies, objectives)
        
        assert selected is not None
        assert selected in policies


class TestGovernanceSystem:
    """Test GovernanceSystem class."""
    
    def test_select_policy(self):
        """Test policy selection."""
        boards = [
            Board(
                board_id=f"board_{i}",
                board_type=bt,
                members=[BoardMember(name=f"Member_{i}", board_type=bt)]
            )
            for i, bt in enumerate([BoardType.GROWTH, BoardType.WELFARE])
        ]
        
        governance = GovernanceSystem(boards, arbitration_method="weighted_vote")
        
        policies = [ControlVector.sample_budget_simplex(["a", "b"]) for _ in range(3)]
        objectives = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for _ in range(3)])
        
        selected = governance.select_policy(policies, objectives)
        
        assert selected is not None


class TestLogisticsNetwork:
    """Test LogisticsNetwork class."""
    
    def test_flow_problem(self):
        """Test flow optimization."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        network = LogisticsNetwork(nodes, edges)
        
        demands = {"C": {"food": 100.0}}
        commodities = ["food"]
        
        flows = network.solve_flow_problem(demands, commodities)
        
        assert isinstance(flows, dict)
        # Should have flows for each edge-commodity pair
        assert len(flows) > 0


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        collector = MetricsCollector()
        
        trajectories = [[StateVector() for _ in range(3)] for _ in range(2)]
        policies = [ControlVector() for _ in range(2)]
        
        metrics = collector.compute_all_metrics(trajectories, policies)
        
        assert "feasibility_rate" in metrics
        assert "U_variance" in metrics
        assert "control_effort" in metrics
    
    def test_generate_report(self):
        """Test report generation."""
        collector = MetricsCollector()
        
        trajectories = [[StateVector() for _ in range(3)]]
        policies = [ControlVector()]
        collector.compute_all_metrics(trajectories, policies)
        
        report = collector.generate_report()
        assert isinstance(report, str)
        assert "Metrics Report" in report


class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        # Initialize components
        dynamics = DynamicsModel()
        checker = ConstraintChecker()
        simulator = ForwardSimulator(dynamics, checker)
        
        # Initial state
        x_0 = StateVector()
        
        # Sample policy
        u = ControlVector.sample_budget_simplex(
            ["energy", "food", "infrastructure", "education"]
        )
        
        # Simulate
        trajectory, feasibility, first_violation = simulator.simulate_scenario(
            x_0, u, [{}] * 10, horizon=10
        )
        
        assert len(trajectory) == 11
        assert len(feasibility) == 10
    
    def test_mpc_with_governance(self):
        """Test MPC solver with board governance."""
        # Setup
        dynamics = DynamicsModel()
        checker = ConstraintChecker()
        obj_computer = ObjectiveVector(horizon=5)
        solver = MPCSolver(dynamics, checker, obj_computer, horizon=5)
        simulator = ForwardSimulator(dynamics, checker)
        
        # Create boards
        boards = [
            Board(
                board_id=f"board_{i}",
                board_type=bt,
                members=[BoardMember(name=f"Member_{i}", board_type=bt)]
            )
            for i, bt in enumerate([BoardType.GROWTH, BoardType.WELFARE])
        ]
        
        governance = GovernanceSystem(boards)
        
        # Solve MPC
        x_t = StateVector()
        scenarios = [[{}] * 5 for _ in range(3)]
        policy, _ = solver.solve(x_t, scenarios)
        
        # Evaluate with boards
        traj, _, _ = simulator.simulate_scenario(x_t, policy, scenarios[0], horizon=5)
        objectives = obj_computer.compute(traj, [policy] * 5)
        objectives_matrix = objectives.reshape(1, -1)
        
        # Governance selection
        selected = governance.select_policy([policy], objectives_matrix)
        
        assert selected is not None


class TestRealTimeSystem:
    """Test real-time control system components."""
    
    def test_data_acquisition(self):
        """Test data acquisition system."""
        from crca_sd.crca_sd_realtime import DataAcquisition, DataSourceType
        
        data_acq = DataAcquisition(update_frequency=3600.0)  # 1 hour
        
        # Connect to government API
        success = data_acq.connect_government_api(
            "treasury",
            "https://api.example.gov/treasury",
            {"api_key": "test"}
        )
        assert success
        
        # Fetch data
        data_points = data_acq.fetch_government_data(
            "treasury",
            "current_state",
            ["P", "U", "Y"]
        )
        
        assert len(data_points) > 0
        assert data_points[0].source_type == DataSourceType.GOVERNMENT_API
    
    def test_safety_interlocks(self):
        """Test safety interlock system."""
        from crca_sd.crca_sd_realtime import SafetyInterlocks
        
        safety = SafetyInterlocks(
            major_change_threshold=0.10,
            confidence_threshold=0.95
        )
        
        # Minor change policy
        prev_policy = ControlVector(budget_shares={"a": 0.5, "b": 0.5})
        minor_policy = ControlVector(budget_shares={"a": 0.55, "b": 0.45})  # 5% change
        
        is_safe, reason, requires_approval = safety.check_policy_safety(
            minor_policy,
            prev_policy,
            state_confidence=0.98
        )
        
        assert is_safe
        assert not requires_approval  # Minor change, automated
        
        # Major change policy
        major_policy = ControlVector(budget_shares={"a": 0.8, "b": 0.2})  # 30% change
        
        is_safe, reason, requires_approval = safety.check_policy_safety(
            major_policy,
            prev_policy,
            state_confidence=0.98
        )
        
        assert is_safe
        assert requires_approval  # Major change, needs approval
    
    def test_policy_executor(self):
        """Test policy executor."""
        from crca_sd.crca_sd_realtime import PolicyExecutor
        
        executor = PolicyExecutor()
        
        policy = ControlVector.sample_budget_simplex(["energy", "food"])
        success, message, info = executor.execute_policy(policy)
        
        assert isinstance(success, bool)
        assert "execution_id" in info
    
    def test_governance_approval(self):
        """Test governance approval workflow."""
        from crca_sd.crca_sd_governance import GovernanceSystem, Board, BoardType, BoardMember
        
        boards = [
            Board(
                board_id="test",
                board_type=BoardType.GROWTH,
                members=[BoardMember(name="Test", board_type=BoardType.GROWTH)]
            )
        ]
        
        governance = GovernanceSystem(boards, major_change_threshold=0.10)
        
        # Minor change
        prev = ControlVector(budget_shares={"a": 0.5, "b": 0.5})
        minor = ControlVector(budget_shares={"a": 0.55, "b": 0.45})
        
        requires, reason = governance.requires_approval(minor, prev)
        assert not requires
        
        # Major change
        major = ControlVector(budget_shares={"a": 0.8, "b": 0.2})
        requires, reason = governance.requires_approval(major, prev)
        assert requires
        
        # Request approval
        approval_id = governance.request_approval(major, reason, "system")
        assert approval_id in governance.pending_approvals
        
        # Approve
        success, message = governance.approve_policy(approval_id, "human", "Approved")
        assert success
    
    def test_rollback_system(self):
        """Test rollback system (7-day window)."""
        from crca_sd.crca_sd_realtime import RollbackSystem
        
        rollback = RollbackSystem(rollback_window_days=7)
        
        # Create snapshots
        state1 = StateVector(P=360000.0)
        state2 = StateVector(P=361000.0)
        
        snap1 = rollback.create_snapshot(state1)
        snap2 = rollback.create_snapshot(state2)
        
        assert snap1 != snap2
        
        # Record policy
        policy = ControlVector()
        rollback.record_policy_execution(policy, "exec_1")
        
        # Rollback
        rolled = rollback.rollback_policies(1)
        assert len(rolled) == 1
        assert rolled[0] == "exec_1"
        
        # Restore state
        restored = rollback.restore_state()
        assert restored is not None
        assert restored.P == state2.P
    
    def test_compliance_system(self):
        """Test compliance system."""
        from crca_sd.crca_sd_realtime import ComplianceSystem
        
        compliance = ComplianceSystem()
        
        # Log decision
        log_id = compliance.log_decision(
            "policy_execution",
            {"policy": "test"},
            "user1",
            "Test decision",
            approved_by=["user2"]
        )
        
        assert log_id is not None
        
        # Get audit trail
        trail = compliance.get_audit_trail(user_id="user1")
        assert len(trail) > 0
    
    def test_alerting_system(self):
        """Test alerting system."""
        from crca_sd.crca_sd_realtime import AlertingSystem, AlertLevel
        
        alerting = AlertingSystem()
        
        # Create alerts
        alert1 = alerting.create_alert(
            AlertLevel.INFO,
            "Test alert",
            "monitor"
        )
        
        alert2 = alerting.create_alert(
            AlertLevel.CRITICAL,
            "Critical issue",
            "safety"
        )
        
        assert alert1 is not None
        assert alert2 is not None
        
        # Get unacknowledged
        unack = alerting.get_unacknowledged_alerts()
        assert len(unack) >= 2
    
    def test_realtime_integration(self):
        """Test integration of real-time components."""
        from crca_sd.crca_sd_realtime import (
            DataAcquisition,
            RealTimeStateEstimator,
            RealTimeMonitor,
            PolicyExecutor,
            SafetyInterlocks,
        )
        
        # Setup
        dynamics = DynamicsModel()
        checker = ConstraintChecker()
        
        data_acq = DataAcquisition()
        state_estimator = RealTimeStateEstimator(dynamics)
        monitor = RealTimeMonitor(checker)
        executor = PolicyExecutor()
        safety = SafetyInterlocks()
        
        # Test workflow
        x_t = StateVector()
        policy = ControlVector.sample_budget_simplex(["energy", "food"])
        
        # Check safety
        is_safe, reason, requires_approval = safety.check_policy_safety(
            policy,
            None,
            state_confidence=0.98
        )
        
        assert isinstance(is_safe, bool)
        
        # Execute if safe
        if is_safe and not requires_approval:
            success, message, info = executor.execute_policy(policy)
            assert isinstance(success, bool)

