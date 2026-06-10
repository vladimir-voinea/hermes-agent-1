"""Message analysis utilities for DCP."""

import logging
from typing import Any, Dict, List

from plugins.context_engine.dcp.types import MessageAnalysis

logger = logging.getLogger(__name__)

# Approximate chars per token for rough estimation
_CHARS_PER_TOKEN = 4

# Content that should never be pruned
_PROTECTED_PATTERNS = [
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]",
    "## Active Task",
    "## In Progress",
    "## Pending User Asks",
    "## Remaining Work",
]


def estimate_tokens(content: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(content) // _CHARS_PER_TOKEN)


def extract_content_text(message: Dict[str, Any]) -> str:
    """Extract plain text content from a message for analysis."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content)


def is_protected_message(message: Dict[str, Any], index: int, head_count: int = 3) -> bool:
    """Determine if a message should be protected from pruning."""
    # Always protect system messages
    if message.get("role") == "system":
        return True
    # Protect head messages (first N non-system messages)
    if index < head_count:
        return True
    # Protect messages containing protected patterns
    text = extract_content_text(message)
    for pattern in _PROTECTED_PATTERNS:
        if pattern in text:
            return True
    return False


def is_tool_call(message: Dict[str, Any]) -> bool:
    """Check if message contains tool calls."""
    return bool(message.get("tool_calls"))


def is_tool_result(message: Dict[str, Any]) -> bool:
    """Check if message is a tool result."""
    return message.get("role") == "tool"


def is_error_message(message: Dict[str, Any]) -> bool:
    """Detect error content in tool results or assistant messages."""
    text = extract_content_text(message)
    error_indicators = [
        "error:",
        "exception:",
        "traceback",
        "failed",
        "failure",
        "not found",
        "permission denied",
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in error_indicators)


def analyze_messages(
    messages: List[Dict[str, Any]],
    head_count: int = 3,
) -> List[MessageAnalysis]:
    """Analyze all messages and return classification."""
    results = []
    for i, msg in enumerate(messages):
        text = extract_content_text(msg)
        analysis = MessageAnalysis(
            index=i,
            message_id=msg.get("id") or msg.get("tool_call_id"),
            role=msg.get("role", ""),
            content_preview=text[:200],
            token_estimate=estimate_tokens(text),
            is_protected=is_protected_message(msg, i, head_count),
            is_tool_call=is_tool_call(msg),
            is_tool_result=is_tool_result(msg),
            is_error=is_error_message(msg),
        )
        results.append(analysis)
    return results
