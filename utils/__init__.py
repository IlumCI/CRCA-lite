"""
CRCA Utilities Module

Provides utility functions and classes for formatting, visualization,
and common operations within the CRCA framework.
"""

try:
    from .formatter import Formatter, MarkdownOutputHandler, formatter
    FORMATTER_AVAILABLE = True
except ImportError:
    Formatter = None
    MarkdownOutputHandler = None
    formatter = None
    FORMATTER_AVAILABLE = False

try:
    from .tui import CorporateSwarmTUI, create_corporate_tui, ViewMode, TUIState
    TUI_AVAILABLE = True
except ImportError:
    CorporateSwarmTUI = None
    create_corporate_tui = None
    ViewMode = None
    TUIState = None
    TUI_AVAILABLE = False

__all__ = [
    "Formatter", "MarkdownOutputHandler", "formatter", "FORMATTER_AVAILABLE",
    "CorporateSwarmTUI", "create_corporate_tui", "ViewMode", "TUIState", "TUI_AVAILABLE"
]
