"""
UAB Leiliona - Comprehensive Autonomous Logistics Corporation

A complete example of a fully autonomous logistics company using CorporateSwarm.

This example demonstrates:
- Complete corporate governance structure for logistics operations
- Board of directors with logistics expertise
- Executive team managing operations, finance, technology
- Department structure (Operations, Finance, Technology, Legal, HR)
- Proposal system for strategic decisions
- Democratic voting on major initiatives
- Risk management for logistics operations
- ESG compliance and sustainability
- AOP integration for agent deployment
- Interactive TUI for management
- Real-world logistics tools and operations

Company: UAB Leiliona
Industry: Logistics and Supply Chain Management
Operations: Freight forwarding, warehousing, last-mile delivery, customs clearance
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import CorporateSwarm
from crca_cg.corposwarm import (
    CorporateSwarm,
    CorporateRole,
    DepartmentType,
    ProposalType,
    BoardCommitteeType,
    MeetingType,
)

# Import utilities
try:
    from utils.tui import CorporateSwarmTUI
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
    logger.warning("TUI not available - install rich: pip install rich")


class LogisticsTools:
    """
    Tools for logistics operations.
    
    These would typically integrate with:
    - Transportation Management Systems (TMS)
    - Warehouse Management Systems (WMS)
    - Customs clearance APIs
    - Carrier APIs (FedEx, UPS, DHL, etc.)
    - Payment systems (including x402)
    - Tracking systems
    """
    
    @staticmethod
    def create_shipment(
        origin: str,
        destination: str,
        weight: float,
        dimensions: Dict[str, float],
        service_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Create a new shipment.
        
        Args:
            origin: Origin address
            destination: Destination address
            weight: Weight in kg
            dimensions: Dict with length, width, height in cm
            service_type: Service type (standard, express, overnight)
            
        Returns:
            Dict with shipment_id, tracking_number, estimated_cost, estimated_delivery
        """
        shipment_id = f"SH{int(time.time())}"
        tracking_number = f"LEI{shipment_id}"
        
        # Simplified cost calculation
        base_cost = 10.0
        weight_cost = weight * 2.0
        distance_cost = 0.5  # Simplified
        service_multiplier = {"standard": 1.0, "express": 1.5, "overnight": 2.0}.get(service_type, 1.0)
        
        estimated_cost = (base_cost + weight_cost + distance_cost) * service_multiplier
        estimated_delivery = datetime.now() + timedelta(days={"standard": 5, "express": 2, "overnight": 1}.get(service_type, 5))
        
        return {
            "shipment_id": shipment_id,
            "tracking_number": tracking_number,
            "origin": origin,
            "destination": destination,
            "weight": weight,
            "dimensions": dimensions,
            "service_type": service_type,
            "estimated_cost": estimated_cost,
            "estimated_delivery": estimated_delivery.isoformat(),
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def track_shipment(tracking_number: str) -> Dict[str, Any]:
        """
        Track a shipment by tracking number.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Dict with current status, location, estimated_delivery
        """
        # Simulated tracking data
        statuses = ["in_transit", "at_warehouse", "out_for_delivery", "delivered"]
        current_status = statuses[int(time.time()) % len(statuses)]
        
        locations = {
            "in_transit": "In Transit - Route A",
            "at_warehouse": "Warehouse A",
            "out_for_delivery": "Out for Delivery",
            "delivered": "Delivered - Final Destination"
        }
        
        return {
            "tracking_number": tracking_number,
            "status": current_status,
            "current_location": locations.get(current_status, "Unknown"),
            "estimated_delivery": (datetime.now() + timedelta(days=2)).isoformat(),
            "last_update": datetime.now().isoformat()
        }
    
    @staticmethod
    def process_customs_clearance(
        shipment_id: str,
        customs_value: float,
        commodity_code: str,
        origin_country: str,
        destination_country: str
    ) -> Dict[str, Any]:
        """
        Process customs clearance for international shipment.
        
        Args:
            shipment_id: Shipment ID
            customs_value: Declared value
            commodity_code: HS code
            origin_country: Origin country code
            destination_country: Destination country code
            
        Returns:
            Dict with clearance_status, duties, taxes, clearance_number
        """
        clearance_number = f"CUS{int(time.time())}"
        
        # Simplified duty calculation
        duty_rate = 0.05 if destination_country != origin_country else 0.0
        duties = customs_value * duty_rate
        taxes = customs_value * 0.20  # VAT
        
        return {
            "clearance_number": clearance_number,
            "shipment_id": shipment_id,
            "status": "cleared",
            "duties": duties,
            "taxes": taxes,
            "total_fees": duties + taxes,
            "cleared_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def optimize_route(
        stops: List[Dict[str, Any]],
        vehicle_capacity: float,
        time_windows: Optional[Dict[str, tuple]] = None
    ) -> Dict[str, Any]:
        """
        Optimize delivery route.
        
        Args:
            stops: List of stops with address, weight, priority
            vehicle_capacity: Maximum vehicle capacity
            time_windows: Optional time windows for deliveries
            
        Returns:
            Dict with optimized_route, total_distance, estimated_time, cost
        """
        # Simplified route optimization
        optimized_route = sorted(stops, key=lambda x: x.get("priority", 0), reverse=True)
        total_distance = len(stops) * 15.0  # km per stop
        estimated_time = total_distance / 50.0  # hours at 50 km/h
        cost = total_distance * 0.5  # cost per km
        
        return {
            "optimized_route": [s["address"] for s in optimized_route],
            "total_stops": len(stops),
            "total_distance_km": total_distance,
            "estimated_time_hours": estimated_time,
            "estimated_cost": cost,
            "vehicle_utilization": sum(s.get("weight", 0) for s in stops) / vehicle_capacity
        }
    
    @staticmethod
    def manage_inventory(
        warehouse_id: str,
        sku: str,
        quantity: int,
        operation: str = "check"
    ) -> Dict[str, Any]:
        """
        Manage warehouse inventory.
        
        Args:
            warehouse_id: Warehouse identifier
            sku: Stock keeping unit
            quantity: Quantity for operation
            operation: Operation type (check, add, remove, reserve)
            
        Returns:
            Dict with current_stock, operation_result, location
        """
        # Simulated inventory
        base_stock = 100
        current_stock = base_stock - (int(time.time()) % 50)
        
        if operation == "add":
            current_stock += quantity
        elif operation == "remove":
            current_stock = max(0, current_stock - quantity)
        elif operation == "reserve":
            current_stock = max(0, current_stock - quantity)
        
        return {
            "warehouse_id": warehouse_id,
            "sku": sku,
            "operation": operation,
            "current_stock": current_stock,
            "available_stock": current_stock,
            "location": f"Aisle {int(time.time()) % 10}, Shelf {int(time.time()) % 5}",
            "last_updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def calculate_shipping_cost(
        origin: str,
        destination: str,
        weight: float,
        volume: float,
        service_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Calculate shipping cost.
        
        Args:
            origin: Origin location
            destination: Destination location
            weight: Weight in kg
            volume: Volume in cubic meters
            service_type: Service type
            
        Returns:
            Dict with base_cost, fuel_surcharge, total_cost, currency
        """
        base_cost = 10.0
        weight_cost = weight * 1.5
        volume_cost = volume * 50.0
        fuel_surcharge = (base_cost + weight_cost) * 0.15
        
        service_multipliers = {
            "standard": 1.0,
            "express": 1.5,
            "overnight": 2.0,
            "economy": 0.8
        }
        
        subtotal = (base_cost + weight_cost + volume_cost) * service_multipliers.get(service_type, 1.0)
        total_cost = subtotal + fuel_surcharge
        
        return {
            "base_cost": base_cost,
            "weight_cost": weight_cost,
            "volume_cost": volume_cost,
            "fuel_surcharge": fuel_surcharge,
            "subtotal": subtotal,
            "total_cost": total_cost,
            "currency": "EUR",
            "service_type": service_type
        }


def create_logistics_corporation() -> CorporateSwarm:
    """
    Create UAB Leiliona - a comprehensive autonomous logistics corporation.
    
    Returns:
        CorporateSwarm instance configured for logistics operations
    """
    logger.info("Creating UAB Leiliona - Autonomous Logistics Corporation")
    
    # Create corporation with all advanced features enabled
    corporation = CorporateSwarm(
        name="UAB Leiliona",
        description="Autonomous logistics and supply chain management corporation",
        max_loops=3,
        corporate_model_name="gpt-4o-mini",
        verbose=True,
        config_data={
            "enable_causal_reasoning": True,
            "enable_quant_analysis": True,
            "enable_crca_sd_governance": True,
            "enable_aop": True,
            "enable_queue_execution": True,
            "enable_democratic_discussion": True,
            "enable_financial_oversight": True,
            "budget_limit": 1000000.0,  # 1M EUR budget
            "decision_threshold": 0.65,
            "default_board_size": 7,
            "default_executive_team_size": 5,
        }
    )
    
    # Add specialized logistics executives
    logger.info("Adding executive team...")
    
    # CEO - Strategic leadership
    ceo_id = corporation.add_member(
        name="Elena Vasiljeva",
        role=CorporateRole.CEO,
        department=DepartmentType.OPERATIONS,
        expertise_areas=["strategic_planning", "logistics", "supply_chain", "business_development"],
        voting_weight=3.0
    )
    
    # CFO - Financial management
    cfo_id = corporation.add_member(
        name="Marius Kazlauskas",
        role=CorporateRole.CFO,
        department=DepartmentType.FINANCE,
        expertise_areas=["finance", "accounting", "cost_optimization", "financial_analysis"],
        voting_weight=2.5
    )
    
    # CTO - Technology and automation
    cto_id = corporation.add_member(
        name="Tomas Petras",
        role=CorporateRole.CTO,
        department=DepartmentType.TECHNOLOGY,
        expertise_areas=["technology", "automation", "systems_integration", "digital_transformation"],
        voting_weight=2.5
    )
    
    # COO - Operations management
    coo_id = corporation.add_member(
        name="Inga Jankauskiene",
        role=CorporateRole.COO,
        department=DepartmentType.OPERATIONS,
        expertise_areas=["operations", "warehousing", "transportation", "process_optimization"],
        voting_weight=2.5
    )
    
    # Chief Logistics Officer
    clo_id = corporation.add_member(
        name="Rokas Stankevicius",
        role=CorporateRole.DEPARTMENT_HEAD,
        department=DepartmentType.OPERATIONS,
        expertise_areas=["logistics", "route_optimization", "fleet_management", "last_mile_delivery"],
        voting_weight=2.0
    )
    
    # Add specialized board members
    logger.info("Adding board of directors...")
    
    # Board Chair - Logistics industry expert
    board_chair_id = corporation.add_member(
        name="Dr. Viktoras Zukauskas",
        role=CorporateRole.BOARD_CHAIR,
        department=DepartmentType.OPERATIONS,
        expertise_areas=["governance", "logistics", "supply_chain", "international_trade"],
        voting_weight=3.5,
        independence_status=True
    )
    
    # Independent directors
    corporation.add_member(
        name="Prof. Ruta Navickiene",
        role=CorporateRole.INDEPENDENT_DIRECTOR,
        department=DepartmentType.FINANCE,
        expertise_areas=["finance", "audit", "risk_management", "compliance"],
        voting_weight=3.0,
        independence_status=True
    )
    
    corporation.add_member(
        name="Dr. Andrius Balciunas",
        role=CorporateRole.INDEPENDENT_DIRECTOR,
        department=DepartmentType.TECHNOLOGY,
        expertise_areas=["technology", "innovation", "digital_transformation", "ai"],
        voting_weight=3.0,
        independence_status=True
    )
    
    corporation.add_member(
        name="Jurate Rimkute",
        role=CorporateRole.INDEPENDENT_DIRECTOR,
        department=DepartmentType.LEGAL,
        expertise_areas=["legal", "compliance", "international_law", "regulatory_affairs"],
        voting_weight=3.0,
        independence_status=True
    )
    
    corporation.add_member(
        name="Kestutis Vaitkus",
        role=CorporateRole.INDEPENDENT_DIRECTOR,
        department=DepartmentType.OPERATIONS,
        expertise_areas=["sustainability", "environmental_compliance", "esg", "green_logistics"],
        voting_weight=3.0,
        independence_status=True
    )
    
    corporation.add_member(
        name="Dalia Stasiuliene",
        role=CorporateRole.INDEPENDENT_DIRECTOR,
        department=DepartmentType.HUMAN_RESOURCES,
        expertise_areas=["hr", "talent_management", "organizational_development", "workplace_safety"],
        voting_weight=3.0,
        independence_status=True
    )
    
    # Add department heads
    logger.info("Adding department heads...")
    
    # Finance Department Head
    finance_head_id = corporation.add_member(
        name="Giedre Matuliene",
        role=CorporateRole.DEPARTMENT_HEAD,
        department=DepartmentType.FINANCE,
        expertise_areas=["finance", "accounting", "budgeting", "financial_reporting"],
        voting_weight=2.0
    )
    
    # Technology Department Head
    tech_head_id = corporation.add_member(
        name="Mindaugas Juska",
        role=CorporateRole.DEPARTMENT_HEAD,
        department=DepartmentType.TECHNOLOGY,
        expertise_areas=["technology", "software_development", "infrastructure", "cybersecurity"],
        voting_weight=2.0
    )
    
    # Legal Department Head
    legal_head_id = corporation.add_member(
        name="Asta Kazlauskiene",
        role=CorporateRole.DEPARTMENT_HEAD,
        department=DepartmentType.LEGAL,
        expertise_areas=["legal", "contracts", "compliance", "regulatory"],
        voting_weight=2.0
    )
    
    # HR Department Head
    hr_head_id = corporation.add_member(
        name="Rasa Didžiuliene",
        role=CorporateRole.DEPARTMENT_HEAD,
        department=DepartmentType.HUMAN_RESOURCES,
        expertise_areas=["hr", "recruitment", "training", "employee_relations"],
        voting_weight=2.0
    )
    
    # Operations Department Head (already added as CLO)
    
    # Update department budgets
    logger.info("Configuring department budgets...")
    for dept_id, dept in corporation.departments.items():
        if dept.department_type == DepartmentType.OPERATIONS:
            dept.budget = 400000.0  # 40% of budget for operations
        elif dept.department_type == DepartmentType.FINANCE:
            dept.budget = 150000.0  # 15% for finance
        elif dept.department_type == DepartmentType.TECHNOLOGY:
            dept.budget = 200000.0  # 20% for technology
        elif dept.department_type == DepartmentType.LEGAL:
            dept.budget = 100000.0  # 10% for legal
        elif dept.department_type == DepartmentType.HUMAN_RESOURCES:
            dept.budget = 100000.0  # 10% for HR
        else:
            dept.budget = 50000.0  # 5% for others
    
    # Add logistics tools to member agents
    logger.info("Adding logistics tools to agents...")
    tools = LogisticsTools()
    
    for member_id, member in corporation.members.items():
        if member.agent:
            # Initialize tools list if needed
            if not hasattr(member.agent, 'tools') or member.agent.tools is None:
                member.agent.tools = []
            
            # Add logistics tools based on role using add_tools method
            if member.role in [CorporateRole.CEO, CorporateRole.COO, CorporateRole.DEPARTMENT_HEAD]:
                if member.department == DepartmentType.OPERATIONS:
                    # Add operational tools
                    try:
                        if hasattr(member.agent, 'add_tools'):
                            member.agent.add_tools([
                                tools.create_shipment,
                                tools.track_shipment,
                                tools.optimize_route,
                                tools.manage_inventory,
                            ])
                        else:
                            member.agent.tools.extend([
                                tools.create_shipment,
                                tools.track_shipment,
                                tools.optimize_route,
                                tools.manage_inventory,
                            ])
                    except Exception as e:
                        logger.warning(f"Failed to add operational tools to {member.name}: {e}")
            
            if member.role == CorporateRole.CFO or member.department == DepartmentType.FINANCE:
                # Add financial tools
                try:
                    if hasattr(member.agent, 'add_tools'):
                        member.agent.add_tools([tools.calculate_shipping_cost])
                    else:
                        member.agent.tools.append(tools.calculate_shipping_cost)
                except Exception as e:
                    logger.warning(f"Failed to add financial tools to {member.name}: {e}")
            
            if member.department == DepartmentType.LEGAL:
                # Add legal/compliance tools
                try:
                    if hasattr(member.agent, 'add_tools'):
                        member.agent.add_tools([tools.process_customs_clearance])
                    else:
                        member.agent.tools.append(tools.process_customs_clearance)
                except Exception as e:
                    logger.warning(f"Failed to add legal tools to {member.name}: {e}")
    
    logger.info(f"Created UAB Leiliona with {len(corporation.members)} members")
    logger.info(f"Board members: {len(corporation.board_members)}")
    logger.info(f"Executive team: {len(corporation.executive_team)}")
    logger.info(f"Departments: {len(corporation.departments)}")
    logger.info(f"Committees: {len(corporation.board_committees)}")
    
    return corporation


def demonstrate_logistics_operations(corporation: CorporateSwarm) -> None:
    """
    Demonstrate various logistics operations and corporate decisions.
    
    Args:
        corporation: CorporateSwarm instance
    """
    logger.info("=== Demonstrating Logistics Operations ===")
    
    # 1. Create a strategic proposal for fleet expansion
    logger.info("\n1. Creating proposal for fleet expansion...")
    proposal_id = corporation.create_proposal(
        title="Fleet Expansion: Add 10 Electric Delivery Vehicles",
        description=(
            "Proposal to expand our delivery fleet with 10 electric vehicles "
            "to support growing demand and improve sustainability metrics. "
            "Total investment: EUR 350,000. Expected ROI: 18% over 3 years. "
            "Reduces carbon footprint by 25% for last-mile deliveries."
        ),
        proposal_type=ProposalType.INVESTMENT,
        sponsor_id=corporation.executive_team[0],  # CEO
        department=DepartmentType.OPERATIONS,
        budget_impact=350000.0,
        timeline="6 months for procurement and deployment"
    )
    
    logger.info(f"Created proposal: {proposal_id}")
    
    # 2. Conduct vote on the proposal
    logger.info("\n2. Conducting corporate vote on fleet expansion...")
    vote = corporation.conduct_corporate_vote(proposal_id)
    logger.info(f"Vote result: {vote.result.value}")
    logger.info(f"Participants: {len(vote.participants)}")
    logger.info(f"Governance consensus: {vote.governance_consensus:.1%}")
    
    if vote.causal_reasoning_summary:
        logger.info(f"Causal reasoning: {vote.causal_reasoning_summary[:200]}...")
    
    if vote.quant_signals:
        logger.info(f"Quantitative signals: {vote.quant_signals}")
    
    # 3. Create proposal for warehouse automation
    logger.info("\n3. Creating proposal for warehouse automation...")
    automation_proposal_id = corporation.create_proposal(
        title="Warehouse Automation System Implementation",
        description=(
            "Implement automated sorting and inventory management system "
            "in main warehouse. Investment: EUR 500,000. "
            "Improves efficiency by 40%, reduces errors by 90%. "
            "Payback period: 2.5 years."
        ),
        proposal_type=ProposalType.INVESTMENT,
        sponsor_id=corporation.executive_team[2],  # CTO
        department=DepartmentType.TECHNOLOGY,
        budget_impact=500000.0,
        timeline="12 months for implementation and training"
    )
    
    # 4. Schedule board meeting
    logger.info("\n4. Scheduling board meeting...")
    meeting_id = corporation.schedule_board_meeting(
        meeting_type=MeetingType.REGULAR_BOARD,
        agenda=[
            "Q4 Financial Review",
            "Fleet Expansion Proposal Discussion",
            "Warehouse Automation Update",
            "ESG Compliance Review",
            "Risk Assessment Update"
        ]
    )
    logger.info(f"Scheduled meeting: {meeting_id}")
    
    # 5. Conduct risk assessment
    logger.info("\n5. Conducting comprehensive risk assessment...")
    risk_assessments = corporation.conduct_risk_assessment("comprehensive")
    logger.info(f"Assessed {len(risk_assessments)} risk categories")
    
    for category, assessment in list(risk_assessments.items())[:5]:
        logger.info(f"  {category}: {assessment.risk_level} (score: {assessment.risk_score:.2%})")
    
    # 6. Calculate ESG score
    logger.info("\n6. Calculating ESG score...")
    esg_score = corporation.calculate_esg_score()
    logger.info(f"Overall ESG: {esg_score.overall_score:.1f}%")
    logger.info(f"  Environmental: {esg_score.environmental_score:.1f}%")
    logger.info(f"  Social: {esg_score.social_score:.1f}%")
    logger.info(f"  Governance: {esg_score.governance_score:.1f}%")
    logger.info(f"  Carbon Footprint: {esg_score.carbon_footprint:.2f} tons CO2")
    
    # 7. Establish compliance framework
    logger.info("\n7. Establishing compliance frameworks...")
    compliance_frameworks = corporation.establish_compliance_framework("comprehensive")
    logger.info(f"Established {len(compliance_frameworks)} compliance frameworks")
    
    for name, framework in compliance_frameworks.items():
        logger.info(f"  {name}: {framework.compliance_status} (score: {framework.compliance_score:.1f}%)")
    
    # 8. Manage stakeholder engagement
    logger.info("\n8. Managing stakeholder engagement...")
    stakeholder_engagements = corporation.manage_stakeholder_engagement("all")
    logger.info(f"Engaged with {len(stakeholder_engagements)} stakeholder groups")
    
    for stype, engagement in stakeholder_engagements.items():
        logger.info(f"  {stype}: Satisfaction {engagement.satisfaction_score:.1f}%")
    
    # 9. Demonstrate logistics operations using tools
    logger.info("\n9. Demonstrating logistics operations...")
    tools = LogisticsTools()
    
    # Create a shipment
    shipment = tools.create_shipment(
        origin="Vilnius, Lithuania",
        destination="Riga, Latvia",
        weight=150.0,
        dimensions={"length": 120, "width": 80, "height": 60},
        service_type="express"
    )
    logger.info(f"Created shipment: {shipment['tracking_number']}")
    logger.info(f"  Cost: EUR {shipment['estimated_cost']:.2f}")
    logger.info(f"  ETA: {shipment['estimated_delivery']}")
    
    # Track shipment
    tracking = tools.track_shipment(shipment['tracking_number'])
    logger.info(f"Tracking: {tracking['status']} at {tracking['current_location']}")
    
    # Process customs clearance
    customs = tools.process_customs_clearance(
        shipment_id=shipment['shipment_id'],
        customs_value=5000.0,
        commodity_code="8708.29",
        origin_country="LT",
        destination_country="LV"
    )
    logger.info(f"Customs cleared: {customs['clearance_number']}")
    logger.info(f"  Duties: EUR {customs['duties']:.2f}, Taxes: EUR {customs['taxes']:.2f}")
    
    # Optimize route
    stops = [
        {"address": "Stop 1", "weight": 50, "priority": 1},
        {"address": "Stop 2", "weight": 30, "priority": 2},
        {"address": "Stop 3", "weight": 40, "priority": 1},
    ]
    route = tools.optimize_route(stops, vehicle_capacity=200.0)
    logger.info(f"Optimized route: {route['total_distance_km']:.1f} km, {route['estimated_time_hours']:.1f} hours")
    
    # 10. Get corporate status
    logger.info("\n10. Corporate Status Summary...")
    status = corporation.get_corporate_status()
    logger.info(f"Total Members: {status['total_members']}")
    logger.info(f"Active Proposals: {status['active_proposals']}")
    logger.info(f"Total Votes: {status['total_votes']}")
    logger.info(f"Board Meetings: {status['board_meetings']}")
    
    logger.info("\n=== Demonstration Complete ===")


def run_logistics_corporation_interactive() -> None:
    """
    Run the logistics corporation with interactive TUI.
    """
    logger.info("Starting UAB Leiliona - Interactive Mode")
    
    # Create corporation
    corporation = create_logistics_corporation()
    
    # Run demonstration
    demonstrate_logistics_operations(corporation)
    
    # Launch TUI if available
    if TUI_AVAILABLE:
        logger.info("\nLaunching Interactive TUI...")
        logger.info("Use the TUI to:")
        logger.info("  - View corporate status and metrics")
        logger.info("  - Create proposals and conduct votes")
        logger.info("  - Schedule and manage board meetings")
        logger.info("  - Monitor ESG scores and risk assessments")
        logger.info("  - Manage stakeholders and compliance")
        logger.info("  - View AOP integration status")
        logger.info("\nPress Q to quit the TUI")
        
        try:
            corporation.launch_tui(refresh_rate=1.0)
        except KeyboardInterrupt:
            logger.info("TUI closed by user")
    else:
        logger.warning("TUI not available. Install rich: pip install rich")
        logger.info("Running in non-interactive mode")
        
        # Show status
        status = corporation.get_corporate_status()
        logger.info("\n=== Corporate Status ===")
        logger.info(f"Name: {status['name']}")
        logger.info(f"Members: {status['total_members']}")
        logger.info(f"Board: {status['board_members']}")
        logger.info(f"Executives: {status['executive_team']}")
        logger.info(f"Departments: {status['departments']}")
        logger.info(f"Committees: {status['board_committees']}")
        logger.info(f"Proposals: {status['active_proposals']}")
        logger.info(f"Votes: {status['total_votes']}")


def run_logistics_corporation_headless() -> None:
    """
    Run the logistics corporation in headless mode (no TUI).
    
    Useful for automated operations, testing, or integration with other systems.
    """
    logger.info("Starting UAB Leiliona - Headless Mode")
    
    # Create corporation
    corporation = create_logistics_corporation()
    
    # Run demonstration
    demonstrate_logistics_operations(corporation)
    
    # Example: Automated decision-making loop
    logger.info("\n=== Running Automated Decision Loop ===")
    
    # Create a proposal for operational improvement
    proposal_id = corporation.create_proposal(
        title="Implement Real-Time Tracking System",
        description=(
            "Deploy real-time GPS tracking for all delivery vehicles. "
            "Improves customer satisfaction and operational efficiency. "
            "Investment: EUR 75,000."
        ),
        proposal_type=ProposalType.OPERATIONAL_CHANGE,
        sponsor_id=corporation.executive_team[2],  # CTO
        department=DepartmentType.TECHNOLOGY,
        budget_impact=75000.0,
        timeline="3 months"
    )
    
    # Conduct vote
    vote = corporation.conduct_corporate_vote(proposal_id)
    
    if vote.result.value == "approved":
        logger.info("Proposal approved! Implementing tracking system...")
        # Here you would execute the actual implementation
    else:
        logger.info(f"Proposal {vote.result.value}, will be reconsidered")
    
    # Get final status
    status = corporation.get_corporate_status()
    logger.info("\n=== Final Corporate Status ===")
    logger.info(f"Total Decisions Made: {status['total_votes']}")
    logger.info(f"Active Proposals: {status['active_proposals']}")
    logger.info(f"ESG Score: {status['esg_governance']['overall_score']:.1f}%")
    
    return corporation


def main():
    """
    Main entry point for the logistics corporation example.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UAB Leiliona - Autonomous Logistics Corporation Example"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "headless", "demo"],
        default="interactive",
        help="Run mode: interactive (with TUI), headless (no TUI), or demo (quick demonstration)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time} | {level} | {message}"
        )
    
    try:
        if args.mode == "interactive":
            run_logistics_corporation_interactive()
        elif args.mode == "headless":
            run_logistics_corporation_headless()
        elif args.mode == "demo":
            # Quick demonstration
            corporation = create_logistics_corporation()
            demonstrate_logistics_operations(corporation)
            logger.info("\nDemo complete. Use --mode interactive for full TUI experience.")
    except KeyboardInterrupt:
        logger.info("\nShutting down UAB Leiliona...")
    except Exception as e:
        logger.error(f"Error running logistics corporation: {e}")
        raise


if __name__ == "__main__":
    main()

