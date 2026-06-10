"""Dynamic Context Pruning (DCP) context engine."""

import json
import logging
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from plugins.context_engine.dcp.compression import compress_ranges
from plugins.context_engine.dcp.message_utils import analyze_messages
from plugins.context_engine.dcp.strategies import deduplicate_tool_results, purge_error_chains

logger = logging.getLogger(__name__)


class DCPContextEngine(ContextEngine):
    """Context engine that prunes conversation context dynamically.

    Strategies applied in order:
    1. Deduplicate identical tool results
    2. Purge failed tool call chains
    3. Compress contiguous ranges of old messages
    """

    threshold_percent: float = 0.70
    protect_first_n: int = 3
    protect_last_n: int = 6

    def __init__(self, quiet_mode: bool = False):
        super().__init__()
        self.quiet_mode = quiet_mode
        self._session_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "dcp"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if prompt_tokens is None:
            prompt_tokens = self.last_prompt_tokens
        return prompt_tokens > self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before API call."""
        if not messages:
            return False
        rough = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return rough > self.threshold_tokens

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if there's anything beyond protected head/tail."""
        if len(messages) <= self.protect_first_n + 3:
            return False
        analyses = analyze_messages(messages, head_count=self.protect_first_n)
        compressible = [a for a in analyses if not a.is_protected]
        return len(compressible) > 3

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        original_count = len(messages)
        analyses = analyze_messages(messages, head_count=self.protect_first_n)

        # Strategy 1: Deduplicate
        messages, analyses = deduplicate_tool_results(messages, analyses)

        # Strategy 2: Error purge
        messages, analyses = purge_error_chains(messages, analyses)

        # Strategy 3: Range compression
        messages, _ = compress_ranges(messages, analyses, focus_topic)

        if not self.quiet_mode:
            logger.info(
                "DCP compressed %d -> %d messages (focus=%s)",
                original_count, len(messages), focus_topic,
            )

        self.compression_count += 1
        return messages

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self.compression_count = 0

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._session_id = None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_id = None

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = self.name
        status["session_id"] = self._session_id
        return status

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "dcp_status",
                "description": "Get the current status of the DCP context engine",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "dcp_compress",
                "description": "Manually trigger DCP compression with an optional focus topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_topic": {
                            "type": "string",
                            "description": "Optional topic to focus compression on",
                        },
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name == "dcp_status":
            return json.dumps(self.get_status())
        if name == "dcp_compress":
            focus = args.get("focus_topic") if args else None
            return json.dumps({"triggered": True, "focus_topic": focus})
        return json.dumps({"error": f"Unknown DCP tool: {name}"})
