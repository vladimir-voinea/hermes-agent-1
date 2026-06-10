"""DCP type definitions."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MessageAnalysis:
    """Analysis result for a single message."""

    index: int
    message_id: Optional[str]
    role: str
    content_preview: str
    token_estimate: int
    is_protected: bool
    is_tool_call: bool
    is_tool_result: bool
    is_error: bool
    is_duplicate: bool = False
    compression_topic: Optional[str] = None
