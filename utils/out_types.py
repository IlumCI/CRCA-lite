"""
Output type definitions for history and conversation formatting.

Defines type literals for specifying output formats when extracting
conversation history or agent responses.
"""

from typing import Literal

HistoryOutputType = Literal[
    "list", "dict", "dictionary", "string", "str", "final", "last",
    "json", "all", "yaml", "xml", "dict-all-except-first",
    "str-all-except-first", "basemodel", "dict-final", "list-final"
]

OutputType = HistoryOutputType
