"""
Rich text formatting and console output utilities.

This module provides classes and functions for formatting and displaying
rich text content in terminal environments using the Rich library.
Supports markdown rendering, syntax highlighting, streaming output,
and interactive dashboards.
"""

import random
import re
import time
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def _choose_random_color() -> str:
    """Select a random color from available Rich color palette.
    
    Returns:
        str: Color name suitable for Rich styling.
    """
    return random.choice(["red", "green", "blue", "yellow", "magenta", "cyan", "white"])


class MarkdownOutputHandler:
    """Handler for rendering markdown content with syntax highlighting.
    
    Processes markdown text, extracts code blocks, and renders them
    with appropriate syntax highlighting using Rich components.
    """
    
    # Compiled regex patterns for log cleaning
    _LOG_PATTERNS = [
        (re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| (INFO|DEBUG|WARNING|ERROR).*?\|.*?\|"), ""),
        (re.compile(r"INFO.*?\|.*?\|.*?\|"), ""),
        (re.compile(r"DEBUG.*?\|.*?\|.*?\|"), ""),
        (re.compile(r"WARNING.*?\|.*?\|.*?\|"), ""),
        (re.compile(r"ERROR.*?\|.*?\|.*?\|"), ""),
    ]
    
    _SPINNER_CHARS = "[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]"
    _SPINNER_PATTERNS = [
        (re.compile(_SPINNER_CHARS), ""),
        (re.compile(rf"{_SPINNER_CHARS} Processing\.\.\."), ""),
        (re.compile(rf"{_SPINNER_CHARS} Loop \d+/\d+"), ""),
    ]
    
    _ARTIFACT_PATTERNS = [
        (re.compile(r"Generated content:"), ""),
        (re.compile(r"Evaluation result:"), ""),
        (re.compile(r"Refined content:"), ""),
    ]
    
    _CODE_BLOCK_PATTERN = re.compile(r"```(?P<lang>\w+)?\n(?P<code>.*?)\n```", re.DOTALL | re.MULTILINE)
    
    def __init__(self, console: Console) -> None:
        """Initialize handler with console instance.
        
        Args:
            console: Rich console instance for rendering output.
        """
        self.console = console
    
    def _clean_output(self, output: str) -> str:
        """Remove log artifacts and normalize whitespace.
        
        Args:
            output: Raw output string to clean.
            
        Returns:
            Cleaned output string with normalized formatting.
        """
        if not output:
            return ""
        
        # Remove log prefixes and timestamps
        for pattern, replacement in self._LOG_PATTERNS:
            output = pattern.sub(replacement, output)
        
        # Remove spinner characters and progress indicators
        for pattern, replacement in self._SPINNER_PATTERNS:
            output = pattern.sub(replacement, output)
        
        # Remove artifact markers
        for pattern, replacement in self._ARTIFACT_PATTERNS:
            output = pattern.sub(replacement, output)
        
        # Normalize whitespace
        output = re.sub(r"\n\s*\n\s*\n", "\n\n", output)
        output = re.sub(r"^\s+", "", output, flags=re.MULTILINE)
        output = re.sub(r"\s+$", "", output, flags=re.MULTILINE)
        
        # Ensure proper markdown header formatting
        lines = output.strip().split("\n")
        if lines and not any(line.strip().startswith("#") for line in lines[:3]):
            first_line = lines[0].strip()
            if (first_line and len(first_line) < 100 and 
                not first_line.startswith(("**", "#", "-", "*", ">", "```")) and
                (not first_line.endswith((",", ".", ":", ";")) or first_line.endswith(":"))):
                output = f"## {first_line}\n\n" + "\n".join(lines[1:])
            else:
                output = "\n".join(lines)
        
        return output.strip()
    
    def _parse_content_parts(self, content: str) -> List[tuple]:
        """Parse content into markdown and code block parts.
        
        Args:
            content: Content string to parse.
            
        Returns:
            List of (type, content) tuples where type is 'markdown' or 'code'.
        """
        parts = []
        current_pos = 0
        
        for match in self._CODE_BLOCK_PATTERN.finditer(content):
            if match.start() > current_pos:
                markdown_content = content[current_pos:match.start()].strip()
                if markdown_content:
                    parts.append(("markdown", markdown_content))
            
            lang = match.group("lang") or "text"
            code = match.group("code")
            parts.append(("code", (lang, code)))
            current_pos = match.end()
        
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                parts.append(("markdown", remaining))
        
        return parts if parts else [("markdown", content)]
    
    def _render_content_parts(self, parts: List[tuple]) -> List[Any]:
        """Render parsed content parts as Rich objects.
        
        Args:
            parts: List of (type, content) tuples from _parse_content_parts.
            
        Returns:
            List of Rich renderable objects.
        """
        rendered = []
        
        for part_type, content in parts:
            if part_type == "markdown":
                try:
                    rendered.append(Markdown(content, code_theme="monokai"))
                except Exception:
                    rendered.append(Text(content, style="white"))
            elif part_type == "code":
                lang, code = content
                try:
                    rendered.append(Syntax(code, lang, theme="monokai", line_numbers=True, word_wrap=True))
                except Exception:
                    rendered.append(Text(f"```{lang}\n{code}\n```", style="white on grey23"))
        
        return rendered
    
    def render_markdown_output(self, content: str, title: str = "", border_style: str = "blue") -> None:
        """Render markdown content with syntax highlighting.
        
        Args:
            content: Markdown content string to render.
            title: Panel title text.
            border_style: Border style for the panel.
        """
        if not content or not content.strip():
            return
        
        cleaned = self._clean_output(content)
        
        try:
            parts = self._parse_content_parts(cleaned)
            rendered = self._render_content_parts(parts)
            
            if rendered:
                self.console.print(Panel(Group(*rendered), title=title, border_style=border_style, padding=(1, 2), expand=False))
            else:
                self.console.print(Panel(Text("No content to display", style="dim italic"), title=title, border_style="yellow"))
        except Exception as e:
            fallback_title = f"{title} [dim](fallback mode)[/dim]" if title else "Content (fallback mode)"
            self.console.print(Panel(cleaned, title=fallback_title, border_style="yellow", subtitle=f"Markdown rendering error: {str(e)}", subtitle_align="left"))


class Formatter:
    """Rich text formatter for console output.
    
    Provides methods for displaying formatted text, tables, progress indicators,
    streaming content, and interactive dashboards in terminal environments.
    """
    
    _SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _STATUS_STYLES = {
        "completed": ("bold green", "✓"),
        "pending": ("bold red", "○"),
        "error": ("bold red", "✗"),
    }
    _PRIORITY_COLORS = {"critical": "red", "high": "yellow", "medium": "blue", "low": "green"}
    _PRIORITY_ICONS = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    
    def __init__(self, md: bool = True) -> None:
        """Initialize formatter with optional markdown support.
        
        Args:
            md: Enable markdown output rendering. Defaults to True.
        """
        self.console = Console()
        self._dashboard_live: Optional[Live] = None
        self._spinner_idx = 0
        self.markdown_handler = MarkdownOutputHandler(self.console) if md else None
    
    def _get_status_text(self, status: str) -> Text:
        """Generate status text with loading animation for running status.
        
        Args:
            status: Status string to format.
            
        Returns:
            Formatted Text object with appropriate styling.
        """
        if status.lower() == "running":
            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER_FRAMES)
            spinner_char = self._SPINNER_FRAMES[self._spinner_idx]
            progress_bar = "█" * (self._spinner_idx % 5) + "░" * (4 - (self._spinner_idx % 5))
            return Text(f"{spinner_char} {status} {progress_bar}", style="bold yellow")
        
        style, symbol = self._STATUS_STYLES.get(status.lower(), ("white", "•"))
        return Text(f"{symbol} {status}", style=style)
    
    def print_panel(self, content: str, title: str = "", style: str = "bold blue") -> None:
        """Display content in a styled panel.
        
        Args:
            content: Content string to display.
            title: Panel title text.
            style: Panel border style.
        """
        if content is None:
            content = "No content to display"
        if not isinstance(content, str):
            content = str(content)
        
        if self.markdown_handler:
            self.markdown_handler.render_markdown_output(content, title, style)
        else:
            try:
                color = _choose_random_color()
                self.console.print(Panel(content, title=title, style=f"bold {color}"))
            except Exception:
                print(f"\n{title}:\n{content}")
    
    def print_markdown(self, content: str, title: str = "", border_style: str = "blue") -> None:
        """Display markdown content with syntax highlighting.
        
        Args:
            content: Markdown content string.
            title: Panel title text.
            border_style: Panel border style.
        """
        if self.markdown_handler:
            self.markdown_handler.render_markdown_output(content, title, border_style)
        else:
            self.print_panel(content, title, border_style)
    
    def print_table(self, title: str, data: Dict[str, List[str]]) -> None:
        """Display data in a formatted table.
        
        Args:
            title: Table title text.
            data: Dictionary mapping categories to lists of items.
        """
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Capabilities", style="green")
        
        for category, items in data.items():
            table.add_row(category, "\n".join(items))
        
        self.console.print(f"\n🔥 {title}:", style="bold yellow")
        self.console.print(table)
    
    def print_progress(self, description: str, task_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute task with progress indicator.
        
        Args:
            description: Progress description text.
            task_fn: Callable to execute.
            *args: Positional arguments for task_fn.
            **kwargs: Keyword arguments for task_fn.
            
        Returns:
            Result from task_fn execution.
        """
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task(description, total=None)
            result = task_fn(*args, **kwargs)
            progress.update(task, completed=True)
        return result
    
    def print_panel_token_by_token(self, tokens: str, title: str = "Output", style: str = "bold cyan", 
                                   delay: float = 0.01, by_word: bool = False) -> None:
        """Display text incrementally, token by token.
        
        Args:
            tokens: Text string to display incrementally.
            title: Panel title text.
            style: Panel border style.
            delay: Delay in seconds between tokens.
            by_word: If True, display by words; otherwise by characters.
        """
        text = Text(style=style)
        token_list = tokens.split() if by_word else tokens
        
        with Live(Panel(text, title=title, border_style=style), console=self.console, refresh_per_second=10) as live:
            for token in token_list:
                text.append(token + (" " if by_word else ""))
                live.update(Panel(text, title=title, border_style=style))
                time.sleep(delay)
    
    def print_streaming_panel(self, streaming_response: Any, title: str = "Agent Streaming Response",
                             style: Optional[str] = None, collect_chunks: bool = False,
                             on_chunk_callback: Optional[Callable] = None) -> str:
        """Display real-time streaming response with live updates.
        
        Args:
            streaming_response: Streaming response generator.
            title: Panel title text.
            style: Panel border style (random if None).
            collect_chunks: Whether to collect individual chunks.
            on_chunk_callback: Optional callback for each chunk.
            
        Returns:
            Complete accumulated response text.
        """
        panel_style = f"bold {_choose_random_color()}" if style is None else style
        
        def _create_panel(text_obj: Text, is_complete: bool = False) -> Panel:
            panel_title = f"[white]{title}[/white]"
            if is_complete:
                panel_title += " [bold green]✅[/bold green]"
            
            display_text = Text.from_markup("")
            display_text.append_text(text_obj)
            if not is_complete:
                display_text.append("▊", style="bold green blink")
            
            return Panel(display_text, title=panel_title, border_style=panel_style, padding=(1, 2), width=self.console.size.width)
        
        streaming_text = Text()
        complete_response = ""
        
        with Live(_create_panel(streaming_text), console=self.console, refresh_per_second=20) as live:
            try:
                for part in streaming_response:
                    if hasattr(part, "choices") and part.choices and part.choices[0].delta.content:
                        chunk = part.choices[0].delta.content
                        streaming_text.append(chunk, style="white")
                        complete_response += chunk
                        
                        if collect_chunks:
                            pass  # Chunks collected in complete_response
                        if on_chunk_callback:
                            on_chunk_callback(chunk)
                        
                        live.update(_create_panel(streaming_text, is_complete=False))
                
                live.update(_create_panel(streaming_text, is_complete=True))
            except Exception as e:
                streaming_text.append(f"\n[Error: {str(e)}]", style="bold red")
                live.update(_create_panel(streaming_text, is_complete=True))
        
        return complete_response
    
    def _create_dashboard_table(self, agents_data: List[Dict[str, Any]], title: str) -> Panel:
        """Create dashboard table with agent status information.
        
        Args:
            agents_data: List of agent information dictionaries.
            title: Dashboard title text.
            
        Returns:
            Panel containing formatted table.
        """
        table = Table(show_header=True, header_style="bold magenta", expand=True, title=title,
                     title_style="bold cyan", border_style="bright_blue", show_lines=True)
        table.add_column("Agent Name", style="cyan", width=30, no_wrap=True)
        table.add_column("Status", style="green", width=20, no_wrap=True)
        table.add_column("Output", style="white", width=100, overflow="fold")
        
        for agent in agents_data:
            table.add_row(Text(agent["name"], style="bold cyan"), self._get_status_text(agent["status"]), Text(str(agent["output"])))
        
        return Panel(table, border_style="bright_blue", padding=(1, 2),
                    title=f"[bold cyan]{title}[/bold cyan] - Total Agents: [bold green]{len(agents_data)}[/bold green]", expand=True)
    
    def print_agent_dashboard(self, agents_data: List[Dict[str, Any]], title: str = "Concurrent Workflow Dashboard",
                             is_final: bool = False) -> None:
        """Display interactive agent dashboard with live updates.
        
        Args:
            agents_data: List of agent information dictionaries.
            title: Dashboard title text.
            is_final: Whether this is the final update.
        """
        if self._dashboard_live is None:
            self._dashboard_live = Live(self._create_dashboard_table(agents_data, title), console=self.console,
                                       refresh_per_second=10, transient=False)
            self._dashboard_live.start()
        else:
            self._dashboard_live.update(self._create_dashboard_table(agents_data, title))
            if is_final:
                self.console.print()
    
    def stop_dashboard(self) -> None:
        """Stop and cleanup dashboard display."""
        if self._dashboard_live is not None:
            self._dashboard_live.stop()
            self.console.print()
            self._dashboard_live = None
    
    def print_plan_tree(self, task_description: str, steps: List[Dict[str, Any]], print_on: bool = True) -> None:
        """Display task plan as hierarchical tree structure.
        
        Args:
            task_description: Main task description text.
            steps: List of step dictionaries with step_id, description, priority, and optional dependencies.
            print_on: Whether to print to console or just log.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        tree = Tree(f"[bold cyan]📋 Plan: {task_description}[/bold cyan]")
        step_nodes = {}
        
        for step in steps:
            step_id = step.get("step_id", "")
            description = step.get("description", "")
            priority = step.get("priority", "medium").lower()
            dependencies = step.get("dependencies", [])
            
            priority_color = self._PRIORITY_COLORS.get(priority, "white")
            priority_icon = self._PRIORITY_ICONS.get(priority, "○")
            
            step_label = f"[{priority_color}]{priority_icon} {step_id}[/{priority_color}]: {description}"
            if dependencies:
                step_label += f" [dim](depends on: {', '.join(dependencies)})[/dim]"
            
            step_nodes[step_id] = tree.add(step_label)
        
        if print_on:
            self.console.print("\n")
            self.console.print(tree)
            self.console.print("")
        else:
            logger.info(f"Plan created: {task_description}")
            for step in steps:
                logger.info(f"  - {step.get('step_id')} ({step.get('priority')}): {step.get('description')}")


# Global formatter instance with markdown disabled by default
formatter = Formatter(md=False)
