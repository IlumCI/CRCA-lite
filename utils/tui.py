"""
CorporateSwarm TUI - Comprehensive Interactive Terminal User Interface

A rich, fully interactive terminal interface for monitoring and controlling
CorporateSwarm governance systems in real-time.

Features:
- Live updating dashboard
- Interactive member management
- Proposal creation and voting
- Board meeting scheduling
- ESG score visualization
- Risk assessment management
- AOP integration control
- Real-time metrics
- Full keyboard navigation and interaction
"""

import time
import threading
import sys
import select
import termios
import tty
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TimeElapsedColumn, ProgressColumn
    )
    from rich.columns import Columns
    from rich.align import Align
    from rich import box
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.rule import Rule
    from rich.spinner import Spinner
    from rich.status import Status
    from rich.prompt import Prompt, Confirm
    from rich.console import Group as RichGroup
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Layout = None
    Panel = None
    Table = None
    Text = None
    Live = None
    Progress = None
    Columns = None
    Align = None
    box = None
    Markdown = None
    Tree = None
    Rule = None
    Spinner = None
    Status = None
    Prompt = None
    Confirm = None
    RichGroup = None

from loguru import logger


class ViewMode(str, Enum):
    """TUI view modes."""
    DASHBOARD = "dashboard"
    MEMBERS = "members"
    PROPOSALS = "proposals"
    VOTES = "votes"
    MEETINGS = "meetings"
    ESG = "esg"
    RISK = "risk"
    AOP = "aop"
    COMMITTEES = "committees"
    DEPARTMENTS = "departments"
    COMMAND = "command"


class InteractionMode(str, Enum):
    """Interaction modes for the TUI."""
    VIEW = "view"
    SELECT = "select"
    CREATE = "create"
    EDIT = "edit"
    DELETE = "delete"
    COMMAND = "command"


@dataclass
class TUIState:
    """State for CorporateSwarm TUI display."""
    corporate_swarm: Optional[Any] = None
    view_mode: ViewMode = ViewMode.DASHBOARD
    interaction_mode: InteractionMode = InteractionMode.VIEW
    selected_member_id: Optional[str] = None
    selected_proposal_id: Optional[str] = None
    selected_committee_id: Optional[str] = None
    selected_vote_id: Optional[str] = None
    selected_meeting_id: Optional[str] = None
    selected_index: int = 0
    list_items: List[Any] = field(default_factory=list)
    auto_refresh: bool = True
    refresh_rate: float = 1.0
    last_update: float = field(default_factory=time.time)
    status_data: Optional[Dict[str, Any]] = None
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    command_history: List[str] = field(default_factory=list)
    current_command: str = ""
    message: Optional[str] = None
    message_style: str = "white"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.status_data is None:
            self.status_data = {}
        if self.alerts is None:
            self.alerts = []
        if self.list_items is None:
            self.list_items = []


class CorporateSwarmTUI:
    """
    Comprehensive Interactive Terminal User Interface for CorporateSwarm.
    
    Provides fully interactive interface with:
    - Corporate overview and metrics
    - Interactive member management
    - Proposal creation and voting
    - Board meeting scheduling
    - ESG score visualization
    - Risk assessment management
    - AOP integration control
    - Real-time alerts and notifications
    - Keyboard navigation and commands
    """
    
    def __init__(
        self, 
        corporate_swarm: Any,
        title: str = "CorporateSwarm Governance System",
        refresh_rate: float = 1.0
    ):
        """
        Initialize CorporateSwarm TUI.
        
        Args:
            corporate_swarm: CorporateSwarm instance to monitor
            title: Title for the TUI
            refresh_rate: Refresh rate in seconds
        """
        if not RICH_AVAILABLE:
            raise ImportError("rich is required for TUI. Install with: pip install rich")
        
        self.corporate_swarm = corporate_swarm
        self.console = Console()
        self.title = title
        self.state = TUIState(
            corporate_swarm=corporate_swarm,
            refresh_rate=refresh_rate
        )
        self.is_running = False
        self.start_time = time.time()
        self._should_quit = False
        self._key_queue = None
        self._keyboard_thread = None
        self._setup_key_handlers()
    
    def _setup_key_handlers(self) -> None:
        """Setup keyboard handlers."""
        self._key_handlers: Dict[str, Callable] = {
            'q': self._quit,
            'Q': self._quit,
            'r': self._refresh,
            'R': self._refresh,
            '1': lambda: self._set_view(ViewMode.DASHBOARD),
            '2': lambda: self._set_view(ViewMode.MEMBERS),
            '3': lambda: self._set_view(ViewMode.PROPOSALS),
            '4': lambda: self._set_view(ViewMode.VOTES),
            '5': lambda: self._set_view(ViewMode.MEETINGS),
            '6': lambda: self._set_view(ViewMode.ESG),
            '7': lambda: self._set_view(ViewMode.RISK),
            '8': lambda: self._set_view(ViewMode.AOP),
            '9': lambda: self._set_view(ViewMode.COMMITTEES),
            '0': lambda: self._set_view(ViewMode.DEPARTMENTS),
            'a': self._toggle_auto_refresh,
            'A': self._toggle_auto_refresh,
            'c': self._enter_command_mode,
            'C': self._enter_command_mode,
            'n': self._create_new_item,
            'N': self._create_new_item,
            's': self._select_item,
            'S': self._select_item,
            'enter': self._activate_selected,
            '\r': self._activate_selected,
            '\n': self._activate_selected,
            'up': self._navigate_up,
            'down': self._navigate_down,
            'k': self._navigate_up,
            'j': self._navigate_down,
            'h': self._navigate_left,
            'l': self._navigate_right,
            'escape': self._exit_interaction,
            '\x1b': self._exit_interaction,
            '?': self._show_help,
        }
    
    def _quit(self) -> None:
        """Quit the TUI."""
        self._should_quit = True
    
    def _refresh(self) -> None:
        """Manually refresh the display."""
        self._update_status()
        self.state.message = "Status refreshed"
        self.state.message_style = "green"
    
    def _set_view(self, mode: ViewMode) -> None:
        """Set the current view mode."""
        self.state.view_mode = mode
        self.state.selected_index = 0
        self.state.interaction_mode = InteractionMode.VIEW
        self._update_list_items()
    
    def _toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh mode."""
        self.state.auto_refresh = not self.state.auto_refresh
        status = "enabled" if self.state.auto_refresh else "disabled"
        self.state.message = f"Auto-refresh {status}"
        self.state.message_style = "green"
    
    def _enter_command_mode(self) -> None:
        """Enter command input mode."""
        self.state.interaction_mode = InteractionMode.COMMAND
        self.state.current_command = ""
    
    def _create_new_item(self) -> None:
        """Create a new item based on current view."""
        if self.state.view_mode == ViewMode.PROPOSALS:
            self._create_proposal_interactive()
        elif self.state.view_mode == ViewMode.MEMBERS:
            self._create_member_interactive()
        elif self.state.view_mode == ViewMode.MEETINGS:
            self._schedule_meeting_interactive()
        else:
            self.state.message = "Cannot create items in this view"
            self.state.message_style = "yellow"
    
    def _select_item(self) -> None:
        """Enter selection mode."""
        if self.state.list_items:
            self.state.interaction_mode = InteractionMode.SELECT
            self.state.message = "Use arrow keys to navigate, Enter to select"
            self.state.message_style = "cyan"
        else:
            self.state.message = "No items to select"
            self.state.message_style = "yellow"
    
    def _activate_selected(self) -> None:
        """Activate the selected item."""
        if self.state.interaction_mode == InteractionMode.SELECT:
            self._show_item_details()
        elif self.state.interaction_mode == InteractionMode.COMMAND:
            self._execute_command()
    
    def _navigate_up(self) -> None:
        """Navigate up in list."""
        if self.state.list_items and self.state.selected_index > 0:
            self.state.selected_index -= 1
    
    def _navigate_down(self) -> None:
        """Navigate down in list."""
        if self.state.list_items and self.state.selected_index < len(self.state.list_items) - 1:
            self.state.selected_index += 1
    
    def _navigate_left(self) -> None:
        """Navigate left (back)."""
        self.state.interaction_mode = InteractionMode.VIEW
        self.state.message = None
    
    def _navigate_right(self) -> None:
        """Navigate right (forward)."""
        if self.state.list_items:
            self._select_item()
    
    def _exit_interaction(self) -> None:
        """Exit current interaction mode."""
        self.state.interaction_mode = InteractionMode.VIEW
        self.state.current_command = ""
        self.state.message = None
    
    def _show_help(self) -> None:
        """Show help information."""
        self.state.message = (
            "Help: [1-0] Views | [N]ew | [S]elect | [C]ommand | [R]efresh | "
            "[A]uto-refresh | [Q]uit | [?] Help"
        )
        self.state.message_style = "cyan"
    
    def _update_list_items(self) -> None:
        """Update list items based on current view."""
        if not self.corporate_swarm:
            self.state.list_items = []
            return
        
        if self.state.view_mode == ViewMode.MEMBERS:
            self.state.list_items = list(self.corporate_swarm.members.values())
        elif self.state.view_mode == ViewMode.PROPOSALS:
            self.state.list_items = self.corporate_swarm.proposals
        elif self.state.view_mode == ViewMode.VOTES:
            self.state.list_items = self.corporate_swarm.votes
        elif self.state.view_mode == ViewMode.MEETINGS:
            self.state.list_items = self.corporate_swarm.board_meetings
        elif self.state.view_mode == ViewMode.COMMITTEES:
            self.state.list_items = list(self.corporate_swarm.board_committees.values())
        elif self.state.view_mode == ViewMode.DEPARTMENTS:
            self.state.list_items = list(self.corporate_swarm.departments.values())
        else:
            self.state.list_items = []
        
        # Reset selection index
        if self.state.selected_index >= len(self.state.list_items):
            self.state.selected_index = max(0, len(self.state.list_items) - 1)
    
    def _show_item_details(self) -> None:
        """Show details for selected item."""
        if not self.state.list_items or self.state.selected_index >= len(self.state.list_items):
            return
        
        selected = self.state.list_items[self.state.selected_index]
        
        if self.state.view_mode == ViewMode.MEMBERS:
            self.state.selected_member_id = selected.member_id
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.PROPOSALS:
            self.state.selected_proposal_id = selected.proposal_id
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.VOTES:
            self.state.selected_vote_id = selected.vote_id
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.MEETINGS:
            self.state.selected_meeting_id = selected.meeting_id
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.COMMITTEES:
            self.state.selected_committee_id = selected.committee_id
            self.state.interaction_mode = InteractionMode.VIEW
    
    def _execute_command(self) -> None:
        """Execute a command."""
        if not self.state.current_command:
            self.state.interaction_mode = InteractionMode.VIEW
            return
        
        cmd = self.state.current_command.strip().lower()
        parts = cmd.split()
        
        if not parts:
            self.state.interaction_mode = InteractionMode.VIEW
            return
        
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if command == "vote" and args:
                # Vote on a proposal
                proposal_id = args[0]
                self._conduct_vote_interactive(proposal_id)
            elif command == "proposal" or command == "create":
                self._create_proposal_interactive()
            elif command == "member" or command == "add":
                self._create_member_interactive()
            elif command == "meeting" or command == "schedule":
                self._schedule_meeting_interactive()
            elif command == "risk":
                self._conduct_risk_assessment()
            elif command == "esg":
                self._calculate_esg()
            elif command == "status":
                self._show_full_status()
            else:
                self.state.message = f"Unknown command: {command}. Type 'help' for commands"
                self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Command error: {str(e)}"
            self.state.message_style = "red"
        finally:
            self.state.current_command = ""
            self.state.interaction_mode = InteractionMode.VIEW
    
    def _create_proposal_interactive(self) -> None:
        """Interactively create a proposal."""
        if not self.corporate_swarm:
            self.state.message = "No CorporateSwarm instance"
            self.state.message_style = "red"
            return
        
        try:
            self.console.clear()
            self.console.print(Panel("Create New Proposal", border_style="cyan", box=box.ROUNDED))
            
            title = Prompt.ask("Proposal Title")
            description = Prompt.ask("Description")
            
            # Show proposal types
            proposal_types = [
                "strategic_initiative", "budget_allocation", "hiring_decision",
                "product_launch", "partnership", "merger_acquisition",
                "policy_change", "investment", "operational_change"
            ]
            
            self.console.print("\nProposal Types:")
            for i, ptype in enumerate(proposal_types, 1):
                self.console.print(f"  {i}. {ptype.replace('_', ' ').title()}")
            
            type_choice = Prompt.ask("Select proposal type (number or name)", default="1")
            
            # Parse type choice
            try:
                type_index = int(type_choice) - 1
                if 0 <= type_index < len(proposal_types):
                    proposal_type_str = proposal_types[type_index]
                else:
                    proposal_type_str = type_choice.lower().replace(" ", "_")
            except ValueError:
                proposal_type_str = type_choice.lower().replace(" ", "_")
            
            # Get department
            dept_choice = Prompt.ask(
                "Department (finance/operations/marketing/technology/legal/hr)",
                default="operations"
            )
            
            # Get budget
            budget_str = Prompt.ask("Budget Impact ($)", default="0")
            try:
                budget = float(budget_str)
            except ValueError:
                budget = 0.0
            
            # Get sponsor (use first executive or board member)
            sponsor_id = None
            if self.corporate_swarm.executive_team:
                sponsor_id = self.corporate_swarm.executive_team[0]
            elif self.corporate_swarm.board_members:
                sponsor_id = self.corporate_swarm.board_members[0]
            
            if not sponsor_id:
                self.state.message = "No sponsor available"
                self.state.message_style = "red"
                return
            
            # Import ProposalType
            from crca_cg.corposwarm import ProposalType, DepartmentType
            
            # Map strings to enums
            proposal_type_map = {
                "strategic_initiative": ProposalType.STRATEGIC_INITIATIVE,
                "budget_allocation": ProposalType.BUDGET_ALLOCATION,
                "hiring_decision": ProposalType.HIRING_DECISION,
                "product_launch": ProposalType.PRODUCT_LAUNCH,
                "partnership": ProposalType.PARTNERSHIP,
                "merger_acquisition": ProposalType.MERGER_ACQUISITION,
                "policy_change": ProposalType.POLICY_CHANGE,
                "investment": ProposalType.INVESTMENT,
                "operational_change": ProposalType.OPERATIONAL_CHANGE,
            }
            
            dept_map = {
                "finance": DepartmentType.FINANCE,
                "operations": DepartmentType.OPERATIONS,
                "marketing": DepartmentType.MARKETING,
                "technology": DepartmentType.TECHNOLOGY,
                "legal": DepartmentType.LEGAL,
                "hr": DepartmentType.HUMAN_RESOURCES,
                "human_resources": DepartmentType.HUMAN_RESOURCES,
            }
            
            proposal_type = proposal_type_map.get(proposal_type_str, ProposalType.STRATEGIC_INITIATIVE)
            department = dept_map.get(dept_choice.lower(), DepartmentType.OPERATIONS)
            
            # Create proposal
            proposal_id = self.corporate_swarm.create_proposal(
                title=title,
                description=description,
                proposal_type=proposal_type,
                sponsor_id=sponsor_id,
                department=department,
                budget_impact=budget
            )
            
            self.state.message = f"Proposal created: {proposal_id}"
            self.state.message_style = "green"
            self._update_list_items()
            
        except KeyboardInterrupt:
            self.state.message = "Proposal creation cancelled"
            self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Error creating proposal: {str(e)}"
            self.state.message_style = "red"
    
    def _create_member_interactive(self) -> None:
        """Interactively create a member."""
        if not self.corporate_swarm:
            self.state.message = "No CorporateSwarm instance"
            self.state.message_style = "red"
            return
        
        try:
            self.console.clear()
            self.console.print(Panel("Add New Member", border_style="cyan", box=box.ROUNDED))
            
            name = Prompt.ask("Member Name")
            
            # Show roles
            from crca_cg.corposwarm import CorporateRole, DepartmentType
            
            roles = [
                "ceo", "cfo", "cto", "coo",
                "board_chair", "board_member", "independent_director",
                "department_head", "manager", "employee"
            ]
            
            self.console.print("\nRoles:")
            for i, role in enumerate(roles, 1):
                self.console.print(f"  {i}. {role.replace('_', ' ').title()}")
            
            role_choice = Prompt.ask("Select role (number or name)", default="employee")
            
            # Parse role
            try:
                role_index = int(role_choice) - 1
                if 0 <= role_index < len(roles):
                    role_str = roles[role_index]
                else:
                    role_str = role_choice.lower().replace(" ", "_")
            except ValueError:
                role_str = role_choice.lower().replace(" ", "_")
            
            # Get department
            dept_choice = Prompt.ask(
                "Department (finance/operations/marketing/technology/legal/hr)",
                default="operations"
            )
            
            # Get expertise
            expertise_str = Prompt.ask("Expertise areas (comma-separated)", default="")
            expertise = [e.strip() for e in expertise_str.split(",") if e.strip()]
            
            # Get voting weight
            weight_str = Prompt.ask("Voting Weight", default="1.0")
            try:
                weight = float(weight_str)
            except ValueError:
                weight = 1.0
            
            # Map to enums
            role_map = {
                "ceo": CorporateRole.CEO,
                "cfo": CorporateRole.CFO,
                "cto": CorporateRole.CTO,
                "coo": CorporateRole.COO,
                "board_chair": CorporateRole.BOARD_CHAIR,
                "board_member": CorporateRole.BOARD_MEMBER,
                "independent_director": CorporateRole.INDEPENDENT_DIRECTOR,
                "department_head": CorporateRole.DEPARTMENT_HEAD,
                "manager": CorporateRole.MANAGER,
                "employee": CorporateRole.EMPLOYEE,
            }
            
            dept_map = {
                "finance": DepartmentType.FINANCE,
                "operations": DepartmentType.OPERATIONS,
                "marketing": DepartmentType.MARKETING,
                "technology": DepartmentType.TECHNOLOGY,
                "legal": DepartmentType.LEGAL,
                "hr": DepartmentType.HUMAN_RESOURCES,
                "human_resources": DepartmentType.HUMAN_RESOURCES,
            }
            
            role = role_map.get(role_str, CorporateRole.EMPLOYEE)
            department = dept_map.get(dept_choice.lower(), DepartmentType.OPERATIONS)
            
            # Create member
            member_id = self.corporate_swarm.add_member(
                name=name,
                role=role,
                department=department,
                expertise_areas=expertise,
                voting_weight=weight
            )
            
            self.state.message = f"Member added: {name} ({member_id})"
            self.state.message_style = "green"
            self._update_list_items()
            
        except KeyboardInterrupt:
            self.state.message = "Member creation cancelled"
            self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Error adding member: {str(e)}"
            self.state.message_style = "red"
    
    def _schedule_meeting_interactive(self) -> None:
        """Interactively schedule a meeting."""
        if not self.corporate_swarm:
            self.state.message = "No CorporateSwarm instance"
            self.state.message_style = "red"
            return
        
        try:
            self.console.clear()
            self.console.print(Panel("Schedule Board Meeting", border_style="cyan", box=box.ROUNDED))
            
            from crca_cg.corposwarm import MeetingType
            
            meeting_types = [
                "regular_board", "special_board", "annual_general",
                "committee_meeting", "executive_session", "emergency_meeting"
            ]
            
            self.console.print("\nMeeting Types:")
            for i, mtype in enumerate(meeting_types, 1):
                self.console.print(f"  {i}. {mtype.replace('_', ' ').title()}")
            
            type_choice = Prompt.ask("Select meeting type (number or name)", default="1")
            
            try:
                type_index = int(type_choice) - 1
                if 0 <= type_index < len(meeting_types):
                    meeting_type_str = meeting_types[type_index]
                else:
                    meeting_type_str = type_choice.lower().replace(" ", "_")
            except ValueError:
                meeting_type_str = type_choice.lower().replace(" ", "_")
            
            meeting_type_map = {
                "regular_board": MeetingType.REGULAR_BOARD,
                "special_board": MeetingType.SPECIAL_BOARD,
                "annual_general": MeetingType.ANNUAL_GENERAL,
                "committee_meeting": MeetingType.COMMITTEE_MEETING,
                "executive_session": MeetingType.EXECUTIVE_SESSION,
                "emergency_meeting": MeetingType.EMERGENCY_MEETING,
            }
            
            meeting_type = meeting_type_map.get(meeting_type_str, MeetingType.REGULAR_BOARD)
            
            # Get agenda items
            agenda_str = Prompt.ask("Agenda items (comma-separated)", default="")
            agenda = [a.strip() for a in agenda_str.split(",") if a.strip()]
            
            # Schedule meeting
            meeting_id = self.corporate_swarm.schedule_board_meeting(
                meeting_type=meeting_type,
                agenda=agenda if agenda else None
            )
            
            self.state.message = f"Meeting scheduled: {meeting_id}"
            self.state.message_style = "green"
            self._update_list_items()
            
        except KeyboardInterrupt:
            self.state.message = "Meeting scheduling cancelled"
            self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Error scheduling meeting: {str(e)}"
            self.state.message_style = "red"
    
    def _conduct_vote_interactive(self, proposal_id: Optional[str] = None) -> None:
        """Interactively conduct a vote."""
        if not self.corporate_swarm:
            self.state.message = "No CorporateSwarm instance"
            self.state.message_style = "red"
            return
        
        try:
            if not proposal_id:
                # Show proposals
                if not self.corporate_swarm.proposals:
                    self.state.message = "No proposals available"
                    self.state.message_style = "yellow"
                    return
                
                self.console.clear()
                self.console.print(Panel("Select Proposal to Vote", border_style="cyan", box=box.ROUNDED))
                
                for i, proposal in enumerate(self.corporate_swarm.proposals[-10:], 1):
                    self.console.print(f"  {i}. {proposal.title} ({proposal.proposal_id[:8]})")
                
                choice = Prompt.ask("Select proposal (number or ID)", default="1")
                
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(self.corporate_swarm.proposals[-10:]):
                        proposal_id = self.corporate_swarm.proposals[-10:][index].proposal_id
                    else:
                        proposal_id = choice
                except ValueError:
                    proposal_id = choice
            
            # Conduct vote
            vote = self.corporate_swarm.conduct_corporate_vote(proposal_id)
            
            self.state.message = f"Vote completed: {vote.result.value.upper()}"
            self.state.message_style = "green"
            self._update_list_items()
            
        except Exception as e:
            self.state.message = f"Error conducting vote: {str(e)}"
            self.state.message_style = "red"
    
    def _conduct_risk_assessment(self) -> None:
        """Conduct risk assessment."""
        if not self.corporate_swarm:
            return
        
        try:
            category = Prompt.ask("Risk category (or 'comprehensive')", default="comprehensive")
            assessments = self.corporate_swarm.conduct_risk_assessment(category)
            self.state.message = f"Risk assessment completed: {len(assessments)} categories"
            self.state.message_style = "green"
            self._update_status()
        except Exception as e:
            self.state.message = f"Error: {str(e)}"
            self.state.message_style = "red"
    
    def _calculate_esg(self) -> None:
        """Calculate ESG score."""
        if not self.corporate_swarm:
            return
        
        try:
            esg_score = self.corporate_swarm.calculate_esg_score()
            self.state.message = f"ESG Score: {esg_score.overall_score:.1f}%"
            self.state.message_style = "green"
            self._update_status()
        except Exception as e:
            self.state.message = f"Error: {str(e)}"
            self.state.message_style = "red"
    
    def _show_full_status(self) -> None:
        """Show full corporate status."""
        self._update_status()
        self.state.view_mode = ViewMode.DASHBOARD
        self.state.message = "Full status displayed"
        self.state.message_style = "green"
    
    def _update_status(self) -> None:
        """Update status data from CorporateSwarm."""
        try:
            if self.corporate_swarm:
                self.state.status_data = self.corporate_swarm.get_corporate_status()
                self.state.last_update = time.time()
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def create_layout(self) -> Layout:
        """Create the main layout structure."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render the header panel."""
        uptime = time.time() - self.start_time
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
        
        last_update_str = datetime.fromtimestamp(self.state.last_update).strftime("%H:%M:%S")
        
        header_text = Text()
        header_text.append(self.title, style="bold cyan")
        header_text.append(" | ", style="dim")
        header_text.append(f"Uptime: {uptime_str}", style="green")
        header_text.append(" | ", style="dim")
        header_text.append(f"Last Update: {last_update_str}", style="yellow")
        header_text.append(" | ", style="dim")
        
        if self.state.auto_refresh:
            header_text.append("Auto-Refresh ON", style="green")
        else:
            header_text.append("Auto-Refresh OFF", style="yellow")
        
        header_text.append(" | ", style="dim")
        header_text.append(f"View: {self.state.view_mode.value.upper()}", style="magenta")
        
        if self.state.interaction_mode != InteractionMode.VIEW:
            header_text.append(" | ", style="dim")
            header_text.append(f"Mode: {self.state.interaction_mode.value.upper()}", style="cyan")
        
        return Panel(
            Align.center(header_text),
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def render_footer(self) -> Layout:
        """Render the footer with controls and messages."""
        footer_layout = Layout()
        footer_layout.split_column(
            Layout(name="controls", size=2),
            Layout(name="message", size=2)
        )
        
        # Controls
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("[Q]uit ", style="red")
        controls.append("[R]efresh ", style="yellow")
        controls.append("[1-0] Views ", style="cyan")
        controls.append("[N]ew ", style="green")
        controls.append("[S]elect ", style="blue")
        controls.append("[C]ommand ", style="magenta")
        controls.append("[A]uto-refresh ", style="green")
        controls.append("[?] Help", style="dim")
        
        footer_layout["controls"].update(
            Panel(Align.center(controls), border_style="blue", box=box.ROUNDED)
        )
        
        # Message/Command line
        if self.state.interaction_mode == InteractionMode.COMMAND:
            cmd_text = Text()
            cmd_text.append("Command: ", style="bold cyan")
            cmd_text.append(self.state.current_command, style="white")
            cmd_text.append("_", style="dim")  # Cursor
            footer_layout["message"].update(
                Panel(cmd_text, border_style="cyan", box=box.ROUNDED)
            )
        elif self.state.message:
            msg_text = Text(self.state.message, style=self.state.message_style)
            footer_layout["message"].update(
                Panel(Align.center(msg_text), border_style=self.state.message_style, box=box.ROUNDED)
            )
        else:
            footer_layout["message"].update(
                Panel("", border_style="dim", box=box.ROUNDED)
            )
        
        return footer_layout
    
    def render_dashboard(self) -> Layout:
        """Render the main dashboard view."""
        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=8),
            Layout(name="metrics", ratio=1),
            Layout(name="recent", ratio=1)
        )
        
        layout["overview"].update(self._render_overview_panel())
        layout["metrics"].update(self._render_metrics_panel())
        layout["recent"].update(self._render_recent_activity_panel())
        
        return layout
    
    def _render_overview_panel(self) -> Panel:
        """Render corporate overview panel."""
        status = self.state.status_data or {}
        
        overview_table = Table.grid(padding=(0, 2))
        overview_table.add_column(style="cyan", justify="right")
        overview_table.add_column(style="white")
        
        overview_table.add_row("Corporation:", status.get("name", "N/A"))
        overview_table.add_row("Members:", str(status.get("total_members", 0)))
        overview_table.add_row("Board Members:", str(status.get("board_members", 0)))
        overview_table.add_row("Executive Team:", str(status.get("executive_team", 0)))
        overview_table.add_row("Departments:", str(status.get("departments", 0)))
        overview_table.add_row("Committees:", str(status.get("board_committees", 0)))
        overview_table.add_row("Active Proposals:", str(status.get("active_proposals", 0)))
        overview_table.add_row("Total Votes:", str(status.get("total_votes", 0)))
        
        return Panel(
            overview_table,
            title="Corporate Overview",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def _render_metrics_panel(self) -> Panel:
        """Render key metrics panel."""
        status = self.state.status_data or {}
        esg = status.get("esg_governance", {})
        risk = status.get("risk_management", {})
        
        metrics_table = Table.grid(padding=(0, 2))
        metrics_table.add_column(style="yellow", justify="right")
        metrics_table.add_column(style="white")
        
        # ESG Scores
        esg_score = esg.get("overall_score", 0)
        metrics_table.add_row(
            "ESG Score:",
            self._colorize_score(esg_score, 70, 85)
        )
        
        metrics_table.add_row(
            "Environmental:",
            self._colorize_score(esg.get("environmental_score", 0), 70, 85)
        )
        
        metrics_table.add_row(
            "Social:",
            self._colorize_score(esg.get("social_score", 0), 70, 85)
        )
        
        metrics_table.add_row(
            "Governance:",
            self._colorize_score(esg.get("governance_score", 0), 70, 85)
        )
        
        metrics_table.add_row("", "")  # Spacer
        
        # Risk Metrics
        total_risks = risk.get("total_risks", 0)
        high_risks = risk.get("high_risks", 0)
        medium_risks = risk.get("medium_risks", 0)
        
        metrics_table.add_row(
            "Total Risks:",
            str(total_risks)
        )
        
        metrics_table.add_row(
            "High Risks:",
            f"[red]{high_risks}[/red]"
        )
        
        metrics_table.add_row(
            "Medium Risks:",
            f"[yellow]{medium_risks}[/yellow]"
        )
        
        return Panel(
            metrics_table,
            title="Key Metrics",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def _render_recent_activity_panel(self) -> Panel:
        """Render recent activity panel."""
        status = self.state.status_data or {}
        recent_decisions = status.get("recent_decisions", [])
        
        if not recent_decisions:
            content = Text("No recent decisions", style="dim")
        else:
            content = Table.grid(padding=(0, 1))
            content.add_column(style="cyan")
            content.add_column(style="white")
            content.add_column(style="dim")
            
            for decision in recent_decisions[-5:]:
                proposal = decision.get("proposal", "Unknown")
                result = decision.get("result", "unknown")
                timestamp = decision.get("timestamp", 0)
                
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else "N/A"
                
                result_style = "green" if result == "approved" else "red" if result == "rejected" else "yellow"
                
                content.add_row(
                    proposal[:30] + ("..." if len(proposal) > 30 else ""),
                    f"[{result_style}]{result.upper()}[/{result_style}]",
                    time_str
                )
        
        return Panel(
            content,
            title="Recent Decisions",
            border_style="green",
            box=box.ROUNDED
        )
    
    def render_members_view(self) -> Panel:
        """Render members view with selection."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        members_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        members_table.add_column("", width=2)  # Selection indicator
        members_table.add_column("Name", style="cyan", width=20)
        members_table.add_column("Role", style="yellow", width=15)
        members_table.add_column("Department", style="green", width=15)
        members_table.add_column("Expertise", style="white", width=25)
        members_table.add_column("Voting Weight", style="magenta", justify="right", width=12)
        
        for idx, (member_id, member) in enumerate(self.corporate_swarm.members.items()):
            expertise_str = ", ".join(member.expertise_areas[:2])
            if len(member.expertise_areas) > 2:
                expertise_str += "..."
            
            # Highlight selected row
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            members_table.add_row(
                selector,
                Text(member.name, style=row_style) if row_style else member.name,
                member.role.value.title(),
                member.department.value.title(),
                expertise_str,
                f"{member.voting_weight:.1f}"
            )
        
        return Panel(
            members_table,
            title="Corporate Members",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def render_member_details(self, member_id: str) -> Panel:
        """Render detailed member information."""
        if not self.corporate_swarm or member_id not in self.corporate_swarm.members:
            return Panel("Member not found", border_style="red")
        
        member = self.corporate_swarm.members[member_id]
        
        details_table = Table.grid(padding=(0, 2))
        details_table.add_column(style="cyan", justify="right", width=20)
        details_table.add_column(style="white")
        
        details_table.add_row("Name:", member.name)
        details_table.add_row("Member ID:", member.member_id)
        details_table.add_row("Role:", member.role.value.title())
        details_table.add_row("Department:", member.department.value.title())
        details_table.add_row("Voting Weight:", f"{member.voting_weight:.1f}")
        details_table.add_row("Independence:", "Yes" if member.independence_status else "No")
        details_table.add_row("", "")
        details_table.add_row("Expertise Areas:", ", ".join(member.expertise_areas) if member.expertise_areas else "None")
        details_table.add_row("Committees:", str(len(member.board_committees)))
        
        if member.agent:
            details_table.add_row("", "")
            details_table.add_row("Agent Status:", "Active")
            if hasattr(member.agent, 'agent_name'):
                details_table.add_row("Agent Name:", member.agent.agent_name)
        
        return Panel(
            details_table,
            title=f"Member Details: {member.name}",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def render_proposals_view(self) -> Panel:
        """Render proposals view with selection."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        proposals_table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        proposals_table.add_column("", width=2)
        proposals_table.add_column("Title", style="cyan", width=30)
        proposals_table.add_column("Type", style="yellow", width=15)
        proposals_table.add_column("Department", style="green", width=15)
        proposals_table.add_column("Budget Impact", style="magenta", justify="right", width=15)
        proposals_table.add_column("Status", style="white", width=12)
        
        for idx, proposal in enumerate(self.corporate_swarm.proposals):
            status_style = "green" if proposal.status == "approved" else "red" if proposal.status == "rejected" else "yellow"
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            proposals_table.add_row(
                selector,
                Text(proposal.title[:28] + ("..." if len(proposal.title) > 28 else ""), style=row_style) if row_style else proposal.title[:28] + ("..." if len(proposal.title) > 28 else ""),
                proposal.proposal_type.value.replace("_", " ").title(),
                proposal.department.value.title(),
                f"${proposal.budget_impact:,.2f}",
                f"[{status_style}]{proposal.status.upper()}[/{status_style}]"
            )
        
        return Panel(
            proposals_table,
            title="Proposals",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def render_proposal_details(self, proposal_id: str) -> Panel:
        """Render detailed proposal information."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        proposal = None
        for p in self.corporate_swarm.proposals:
            if p.proposal_id == proposal_id:
                proposal = p
                break
        
        if not proposal:
            return Panel("Proposal not found", border_style="red")
        
        details_table = Table.grid(padding=(0, 2))
        details_table.add_column(style="cyan", justify="right", width=20)
        details_table.add_column(style="white")
        
        details_table.add_row("Title:", proposal.title)
        details_table.add_row("Proposal ID:", proposal.proposal_id)
        details_table.add_row("Type:", proposal.proposal_type.value.replace("_", " ").title())
        details_table.add_row("Department:", proposal.department.value.title())
        details_table.add_row("Budget Impact:", f"${proposal.budget_impact:,.2f}")
        details_table.add_row("Status:", proposal.status.upper())
        details_table.add_row("Timeline:", proposal.timeline or "Not specified")
        details_table.add_row("", "")
        details_table.add_row("Description:", proposal.description[:200] + ("..." if len(proposal.description) > 200 else ""))
        
        if proposal.causal_analysis:
            details_table.add_row("", "")
            details_table.add_row("Causal Analysis:", "Available")
        
        if proposal.quant_analysis:
            details_table.add_row("Quantitative Analysis:", "Available")
        
        if proposal.board_evaluations:
            details_table.add_row("Board Evaluations:", "Available")
        
        return Panel(
            details_table,
            title=f"Proposal Details: {proposal.title}",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def render_votes_view(self) -> Panel:
        """Render votes view with selection."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        votes_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        votes_table.add_column("", width=2)
        votes_table.add_column("Proposal", style="cyan", width=30)
        votes_table.add_column("Participants", style="yellow", justify="right", width=12)
        votes_table.add_column("Result", style="white", width=15)
        votes_table.add_column("Consensus", style="magenta", justify="right", width=12)
        votes_table.add_column("Time", style="dim", width=10)
        
        for idx, vote in enumerate(self.corporate_swarm.votes[-10:]):
            result_style = "green" if vote.result.value == "approved" else "red" if vote.result.value == "rejected" else "yellow"
            consensus = vote.governance_consensus * 100
            
            time_str = datetime.fromtimestamp(vote.timestamp).strftime("%H:%M:%S") if vote.timestamp else "N/A"
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            votes_table.add_row(
                selector,
                Text(vote.proposal.title[:28] + ("..." if len(vote.proposal.title) > 28 else ""), style=row_style) if row_style else vote.proposal.title[:28] + ("..." if len(vote.proposal.title) > 28 else ""),
                str(len(vote.participants)),
                f"[{result_style}]{vote.result.value.upper()}[/{result_style}]",
                f"{consensus:.1f}%",
                time_str
            )
        
        return Panel(
            votes_table,
            title="Voting History",
            border_style="green",
            box=box.ROUNDED
        )
    
    def render_vote_details(self, vote_id: str) -> Panel:
        """Render detailed vote information."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        vote = None
        for v in self.corporate_swarm.votes:
            if v.vote_id == vote_id:
                vote = v
                break
        
        if not vote:
            return Panel("Vote not found", border_style="red")
        
        details_table = Table.grid(padding=(0, 2))
        details_table.add_column(style="cyan", justify="right", width=20)
        details_table.add_column(style="white")
        
        details_table.add_row("Proposal:", vote.proposal.title)
        details_table.add_row("Vote ID:", vote.vote_id)
        details_table.add_row("Result:", vote.result.value.upper())
        details_table.add_row("Participants:", str(len(vote.participants)))
        details_table.add_row("Consensus:", f"{vote.governance_consensus * 100:.1f}%")
        details_table.add_row("Timestamp:", datetime.fromtimestamp(vote.timestamp).strftime("%Y-%m-%d %H:%M:%S") if vote.timestamp else "N/A")
        
        if vote.causal_reasoning_summary:
            details_table.add_row("", "")
            details_table.add_row("Causal Reasoning:", vote.causal_reasoning_summary[:100] + ("..." if len(vote.causal_reasoning_summary) > 100 else ""))
        
        if vote.quant_signals:
            details_table.add_row("Quant Signals:", f"{len(vote.quant_signals)} signals")
        
        # Individual votes summary
        if vote.individual_votes:
            details_table.add_row("", "")
            details_table.add_row("Individual Votes:", "")
            approve_count = sum(1 for v in vote.individual_votes.values() if v.get("vote") == "APPROVE")
            reject_count = sum(1 for v in vote.individual_votes.values() if v.get("vote") == "REJECT")
            abstain_count = sum(1 for v in vote.individual_votes.values() if v.get("vote") == "ABSTAIN")
            
            details_table.add_row("  Approve:", f"[green]{approve_count}[/green]")
            details_table.add_row("  Reject:", f"[red]{reject_count}[/red]")
            details_table.add_row("  Abstain:", f"[yellow]{abstain_count}[/yellow]")
        
        return Panel(
            details_table,
            title=f"Vote Details: {vote.proposal.title}",
            border_style="green",
            box=box.ROUNDED
        )
    
    def render_meetings_view(self) -> Panel:
        """Render board meetings view with selection."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        meetings_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        meetings_table.add_column("", width=2)
        meetings_table.add_column("Type", style="cyan", width=20)
        meetings_table.add_column("Date", style="yellow", width=15)
        meetings_table.add_column("Attendees", style="green", justify="right", width=12)
        meetings_table.add_column("Quorum", style="white", width=10)
        meetings_table.add_column("Resolutions", style="magenta", justify="right", width=12)
        
        for idx, meeting in enumerate(self.corporate_swarm.board_meetings[-10:]):
            date_str = datetime.fromtimestamp(meeting.date).strftime("%Y-%m-%d %H:%M") if meeting.date else "N/A"
            quorum_style = "green" if meeting.quorum_met else "red"
            quorum_symbol = "YES" if meeting.quorum_met else "NO"
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            meetings_table.add_row(
                selector,
                Text(meeting.meeting_type.value.replace("_", " ").title(), style=row_style) if row_style else meeting.meeting_type.value.replace("_", " ").title(),
                date_str,
                str(len(meeting.attendees)),
                f"[{quorum_style}]{quorum_symbol}[/{quorum_style}]",
                str(len(meeting.resolutions))
            )
        
        return Panel(
            meetings_table,
            title="Board Meetings",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def render_meeting_details(self, meeting_id: str) -> Panel:
        """Render detailed meeting information."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        meeting = None
        for m in self.corporate_swarm.board_meetings:
            if m.meeting_id == meeting_id:
                meeting = m
                break
        
        if not meeting:
            return Panel("Meeting not found", border_style="red")
        
        details_table = Table.grid(padding=(0, 2))
        details_table.add_column(style="cyan", justify="right", width=20)
        details_table.add_column(style="white")
        
        details_table.add_row("Meeting ID:", meeting.meeting_id)
        details_table.add_row("Type:", meeting.meeting_type.value.replace("_", " ").title())
        details_table.add_row("Date:", datetime.fromtimestamp(meeting.date).strftime("%Y-%m-%d %H:%M:%S") if meeting.date else "N/A")
        details_table.add_row("Location:", meeting.location or "Virtual")
        details_table.add_row("Attendees:", str(len(meeting.attendees)))
        details_table.add_row("Quorum Met:", "Yes" if meeting.quorum_met else "No")
        details_table.add_row("Resolutions:", str(len(meeting.resolutions)))
        
        if meeting.agenda:
            details_table.add_row("", "")
            details_table.add_row("Agenda:", "")
            for item in meeting.agenda[:5]:
                details_table.add_row("", f"  • {item}")
        
        if meeting.resolutions:
            details_table.add_row("", "")
            details_table.add_row("Resolutions:", "")
            for resolution in meeting.resolutions[:5]:
                details_table.add_row("", f"  • {resolution[:60]}")
        
        if meeting.minutes:
            details_table.add_row("", "")
            details_table.add_row("Minutes:", meeting.minutes[:200] + ("..." if len(meeting.minutes) > 200 else ""))
        
        return Panel(
            details_table,
            title=f"Meeting Details: {meeting.meeting_type.value.replace('_', ' ').title()}",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def render_esg_view(self) -> Panel:
        """Render ESG dashboard view."""
        status = self.state.status_data or {}
        esg = status.get("esg_governance", {})
        
        esg_layout = Layout()
        esg_layout.split_column(
            Layout(name="scores", size=12),
            Layout(name="details", ratio=1)
        )
        
        # Scores with progress bars
        scores_table = Table.grid(padding=(0, 2))
        scores_table.add_column(style="cyan", width=20)
        scores_table.add_column(style="white", width=50)
        scores_table.add_column(style="yellow", width=10)
        
        overall = esg.get("overall_score", 0)
        env = esg.get("environmental_score", 0)
        social = esg.get("social_score", 0)
        gov = esg.get("governance_score", 0)
        
        scores_table.add_row(
            "Overall ESG:",
            self._create_progress_bar(overall, 100),
            f"{overall:.1f}%"
        )
        
        scores_table.add_row(
            "Environmental:",
            self._create_progress_bar(env, 100),
            f"{env:.1f}%"
        )
        
        scores_table.add_row(
            "Social:",
            self._create_progress_bar(social, 100),
            f"{social:.1f}%"
        )
        
        scores_table.add_row(
            "Governance:",
            self._create_progress_bar(gov, 100),
            f"{gov:.1f}%"
        )
        
        esg_layout["scores"].update(
            Panel(scores_table, title="ESG Scores", border_style="green", box=box.ROUNDED)
        )
        
        # Details
        details_table = Table.grid(padding=(0, 2))
        details_table.add_column(style="cyan", justify="right")
        details_table.add_column(style="white")
        
        details_table.add_row("Carbon Footprint:", f"{esg.get('carbon_footprint', 0):.2f} tons CO2")
        details_table.add_row("Diversity Index:", f"{esg.get('diversity_index', 0):.2%}")
        details_table.add_row("Stakeholder Satisfaction:", f"{esg.get('stakeholder_satisfaction', 0):.1f}%")
        
        goals = esg.get("sustainability_goals", [])
        if goals:
            details_table.add_row("", "")
            details_table.add_row("Sustainability Goals:", "")
            for goal in goals[:3]:
                details_table.add_row("", f"  • {goal}")
        
        esg_layout["details"].update(
            Panel(details_table, title="Details", border_style="green", box=box.ROUNDED)
        )
        
        return Panel(
            esg_layout,
            title="ESG Dashboard",
            border_style="green",
            box=box.ROUNDED
        )
    
    def render_risk_view(self) -> Panel:
        """Render risk assessment view."""
        status = self.state.status_data or {}
        risk = status.get("risk_management", {})
        
        if not self.corporate_swarm or not self.corporate_swarm.risk_assessments:
            return Panel("No risk assessments available. Use command 'risk' to conduct assessment.", border_style="yellow")
        
        risk_table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
        risk_table.add_column("Category", style="cyan", width=20)
        risk_table.add_column("Level", style="white", width=12)
        risk_table.add_column("Probability", style="yellow", justify="right", width=12)
        risk_table.add_column("Impact", style="magenta", justify="right", width=12)
        risk_table.add_column("Score", style="red", justify="right", width=12)
        risk_table.add_column("Owner", style="green", width=15)
        
        for risk_id, assessment in list(self.corporate_swarm.risk_assessments.items())[:15]:
            level_style = "red" if assessment.risk_level == "high" else "yellow" if assessment.risk_level == "medium" else "green"
            
            risk_table.add_row(
                assessment.risk_category.title(),
                f"[{level_style}]{assessment.risk_level.upper()}[/{level_style}]",
                f"{assessment.probability:.2%}",
                f"{assessment.impact:.2%}",
                f"{assessment.risk_score:.2%}",
                assessment.owner
            )
        
        return Panel(
            risk_table,
            title="Risk Assessment",
            border_style="red",
            box=box.ROUNDED
        )
    
    def render_aop_view(self) -> Panel:
        """Render AOP integration view."""
        status = self.state.status_data or {}
        aop = status.get("aop_integration", {})
        
        aop_layout = Layout()
        aop_layout.split_column(
            Layout(name="status", size=10),
            Layout(name="queues", ratio=1)
        )
        
        # Status
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style="cyan", justify="right", width=25)
        status_table.add_column(style="white")
        
        status_table.add_row(
            "AOP Enabled:",
            "Yes" if aop.get("enabled", False) else "No"
        )
        
        status_table.add_row(
            "Queue Execution:",
            "Yes" if aop.get("queue_execution_enabled", False) else "No"
        )
        
        status_table.add_row(
            "AOP Server Active:",
            "Yes" if aop.get("aop_server_active", False) else "No"
        )
        
        status_table.add_row(
            "Member Queues:",
            str(aop.get("total_member_queues", 0))
        )
        
        server_info = aop.get("aop_server", {})
        if server_info:
            status_table.add_row("", "")
            status_table.add_row("Server Port:", str(server_info.get("port", "N/A")))
            status_table.add_row("Total Agents:", str(server_info.get("total_agents", 0)))
        
        aop_layout["status"].update(
            Panel(status_table, title="AOP Status", border_style="magenta", box=box.ROUNDED)
        )
        
        # Queue Stats
        queue_stats = aop.get("member_queue_stats", {})
        if queue_stats:
            queues_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            queues_table.add_column("Member", style="cyan", width=20)
            queues_table.add_column("Total Tasks", style="yellow", justify="right", width=12)
            queues_table.add_column("Completed", style="green", justify="right", width=12)
            queues_table.add_column("Pending", style="yellow", justify="right", width=12)
            queues_table.add_column("Status", style="white", width=12)
            
            for member_name, stats in list(queue_stats.items())[:10]:
                queues_table.add_row(
                    member_name[:18] + ("..." if len(member_name) > 18 else ""),
                    str(stats.get("total_tasks", 0)),
                    str(stats.get("completed_tasks", 0)),
                    str(stats.get("pending_tasks", 0)),
                    stats.get("queue_status", "unknown")
                )
            
            aop_layout["queues"].update(
                Panel(queues_table, title="Queue Statistics", border_style="magenta", box=box.ROUNDED)
            )
        else:
            aop_layout["queues"].update(
                Panel("No queue statistics available", border_style="dim", box=box.ROUNDED)
            )
        
        return Panel(
            aop_layout,
            title="AOP Integration",
            border_style="magenta",
            box=box.ROUNDED
        )
    
    def render_committees_view(self) -> Panel:
        """Render board committees view."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        committees_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        committees_table.add_column("Name", style="cyan", width=25)
        committees_table.add_column("Type", style="yellow", width=20)
        committees_table.add_column("Chair", style="green", width=20)
        committees_table.add_column("Members", style="white", justify="right", width=10)
        
        for committee_id, committee in self.corporate_swarm.board_committees.items():
            chair_name = "Unknown"
            if committee.chair and committee.chair in self.corporate_swarm.members:
                chair_name = self.corporate_swarm.members[committee.chair].name
            
            committees_table.add_row(
                committee.name,
                committee.committee_type.value.replace("_", " ").title(),
                chair_name[:18] + ("..." if len(chair_name) > 18 else ""),
                str(len(committee.members))
            )
        
        return Panel(
            committees_table,
            title="Board Committees",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def render_departments_view(self) -> Panel:
        """Render departments view."""
        if not self.corporate_swarm:
            return Panel("No CorporateSwarm instance", border_style="red")
        
        dept_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        dept_table.add_column("Name", style="cyan", width=25)
        dept_table.add_column("Type", style="yellow", width=20)
        dept_table.add_column("Head", style="green", width=20)
        dept_table.add_column("Budget", style="magenta", justify="right", width=15)
        dept_table.add_column("Members", style="white", justify="right", width=10)
        
        for dept_id, dept in self.corporate_swarm.departments.items():
            head_name = "Vacant"
            if dept.head and dept.head in self.corporate_swarm.members:
                head_name = self.corporate_swarm.members[dept.head].name
            
            dept_table.add_row(
                dept.name,
                dept.department_type.value.replace("_", " ").title(),
                head_name[:18] + ("..." if len(head_name) > 18 else ""),
                f"${dept.budget:,.2f}",
                str(len(dept.members))
            )
        
        return Panel(
            dept_table,
            title="Departments",
            border_style="green",
            box=box.ROUNDED
        )
    
    def render(self) -> Layout:
        """Render the complete TUI layout."""
        layout = self.create_layout()
        
        layout["header"].update(self.render_header())
        layout["footer"].update(self.render_footer())
        
        # Render main content based on view mode
        if self.state.view_mode == ViewMode.DASHBOARD:
            main_content = self.render_dashboard()
            layout["left"].update(main_content["overview"])
            layout["right"].split_column(
                Layout(main_content["metrics"]),
                Layout(main_content["recent"])
            )
        elif self.state.view_mode == ViewMode.MEMBERS:
            layout["left"].update(self.render_members_view())
            if self.state.selected_member_id:
                layout["right"].update(self.render_member_details(self.state.selected_member_id))
            else:
                layout["right"].update(
                    Panel(
                        "Select a member to view details\n\nPress [S] to enter selection mode\nPress [Enter] to view details",
                        border_style="dim",
                        box=box.ROUNDED
                    )
                )
        elif self.state.view_mode == ViewMode.PROPOSALS:
            layout["left"].update(self.render_proposals_view())
            if self.state.selected_proposal_id:
                layout["right"].update(self.render_proposal_details(self.state.selected_proposal_id))
            else:
                layout["right"].update(
                    Panel(
                        "Select a proposal to view details\n\nPress [S] to enter selection mode\nPress [N] to create new proposal\nPress [Enter] to view details",
                        border_style="dim",
                        box=box.ROUNDED
                    )
                )
        elif self.state.view_mode == ViewMode.VOTES:
            layout["left"].update(self.render_votes_view())
            if self.state.selected_vote_id:
                layout["right"].update(self.render_vote_details(self.state.selected_vote_id))
            else:
                layout["right"].update(
                    Panel(
                        "Select a vote to view details\n\nPress [S] to enter selection mode\nPress [Enter] to view details",
                        border_style="dim",
                        box=box.ROUNDED
                    )
                )
        elif self.state.view_mode == ViewMode.MEETINGS:
            layout["left"].update(self.render_meetings_view())
            if self.state.selected_meeting_id:
                layout["right"].update(self.render_meeting_details(self.state.selected_meeting_id))
            else:
                layout["right"].update(
                    Panel(
                        "Select a meeting to view details\n\nPress [S] to enter selection mode\nPress [N] to schedule new meeting\nPress [Enter] to view details",
                        border_style="dim",
                        box=box.ROUNDED
                    )
                )
        elif self.state.view_mode == ViewMode.ESG:
            layout["left"].update(self.render_esg_view())
            layout["right"].update(
                Panel(
                    "ESG Dashboard\n\nUse command 'esg' to recalculate scores",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.RISK:
            layout["left"].update(self.render_risk_view())
            layout["right"].update(
                Panel(
                    "Risk Assessment\n\nUse command 'risk' to conduct new assessment",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.AOP:
            layout["left"].update(self.render_aop_view())
            layout["right"].update(
                Panel(
                    "AOP Integration Status\n\nMonitor queue statistics and server status",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.COMMITTEES:
            layout["left"].update(self.render_committees_view())
            layout["right"].update(
                Panel(
                    "Board Committees\n\nView committee structure and membership",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.DEPARTMENTS:
            layout["left"].update(self.render_departments_view())
            layout["right"].update(
                Panel(
                    "Departments\n\nView department structure and budgets",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        
        return layout
    
    def _colorize_score(self, score: float, low_threshold: float = 70, high_threshold: float = 85) -> Text:
        """Colorize a score based on thresholds."""
        if score >= high_threshold:
            style = "green"
        elif score >= low_threshold:
            style = "yellow"
        else:
            style = "red"
        
        return Text(f"{score:.1f}%", style=style)
    
    def _create_progress_bar(self, value: float, max_value: float = 100) -> Text:
        """Create a text-based progress bar."""
        if max_value == 0:
            return Text("N/A", style="dim")
        
        percentage = min(100, max(0, (value / max_value) * 100))
        bar_width = 40
        filled = int((percentage / 100) * bar_width)
        empty = bar_width - filled
        
        if percentage >= 85:
            color = "green"
        elif percentage >= 70:
            color = "yellow"
        else:
            color = "red"
        
        bar = "█" * filled + "░" * empty
        return Text(bar, style=color)
    
    def _setup_keyboard_listener(self) -> None:
        """Setup keyboard listener thread."""
        if sys.platform != "win32":
            try:
                # Save terminal settings
                self._old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self._old_terminal_settings = None
        
        self._key_queue = []
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()
    
    def _keyboard_listener(self) -> None:
        """Keyboard listener thread."""
        try:
            while self.is_running and not self._should_quit:
                key = self._get_key()
                if key:
                    # Handle escape sequences for arrow keys
                    if key == '\x1b':
                        try:
                            if sys.platform != "win32":
                                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                                    seq = sys.stdin.read(2)
                                    if seq == '[A':
                                        key = 'up'
                                    elif seq == '[B':
                                        key = 'down'
                                    elif seq == '[C':
                                        key = 'right'
                                    elif seq == '[D':
                                        key = 'left'
                                    else:
                                        key = 'escape'
                                else:
                                    key = 'escape'
                            else:
                                key = 'escape'
                        except Exception:
                            key = 'escape'
                    
                    self._key_queue.append(key)
                time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Keyboard listener error: {e}")
    
    def _get_key(self) -> Optional[str]:
        """Get a key press from stdin (non-blocking)."""
        if sys.platform == "win32":
            try:
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore')
                    return key
            except Exception:
                pass
        else:
            try:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    return key
            except Exception:
                pass
        return None
    
    def _process_keys(self) -> None:
        """Process queued key presses."""
        if not self._key_queue:
            return
        
        while self._key_queue:
            key = self._key_queue.pop(0)
            
            # Handle command mode
            if self.state.interaction_mode == InteractionMode.COMMAND:
                if key == '\r' or key == '\n':
                    self._activate_selected()
                elif key == '\x1b' or key == 'escape':
                    self._exit_interaction()
                elif key == '\x7f' or key == '\b':  # Backspace
                    if self.state.current_command:
                        self.state.current_command = self.state.current_command[:-1]
                elif len(key) == 1 and key.isprintable() and ord(key) >= 32:
                    self.state.current_command += key
                continue
            
            # Handle normal key presses
            handler = self._key_handlers.get(key)
            if handler:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Error handling key {key}: {e}")
    
    def run_live(
        self,
        update_callback: Optional[Callable] = None,
        refresh_rate: Optional[float] = None
    ) -> None:
        """
        Run the TUI with live updates and keyboard interaction.
        
        Args:
            update_callback: Optional callback function to call before each update
            refresh_rate: Optional refresh rate override
        """
        if refresh_rate:
            self.state.refresh_rate = refresh_rate
        
        self.is_running = True
        self._should_quit = False
        
        # Setup keyboard listener
        self._setup_keyboard_listener()
        
        try:
            with Live(
                self.render(),
                refresh_per_second=1.0 / self.state.refresh_rate,
                screen=True
            ) as live:
                while not self._should_quit:
                    # Process keyboard input
                    self._process_keys()
                    
                    if update_callback:
                        update_callback(self)
                    
                    if self.state.auto_refresh:
                        self._update_status()
                    
                    # Clear message after a delay
                    if self.state.message and time.time() - self.state.last_update > 3:
                        self.state.message = None
                    
                    live.update(self.render())
                    time.sleep(self.state.refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            if sys.platform != "win32" and hasattr(self, '_old_terminal_settings'):
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_terminal_settings)
                except:
                    pass
            self.console.clear()
    
    def run_once(self) -> None:
        """Render the TUI once without live updates."""
        self._update_status()
        self.console.print(self.render())


def create_corporate_tui(corporate_swarm: Any, **kwargs) -> CorporateSwarmTUI:
    """
    Create a CorporateSwarm TUI instance.
    
    Args:
        corporate_swarm: CorporateSwarm instance
        **kwargs: Additional arguments for TUI initialization
        
    Returns:
        CorporateSwarmTUI instance
    """
    return CorporateSwarmTUI(corporate_swarm, **kwargs)
