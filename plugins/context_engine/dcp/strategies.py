"""DCP pruning strategies."""

import logging
from typing import Any, Dict, List, Set, Tuple

from plugins.context_engine.dcp.message_utils import extract_content_text, is_tool_result
from plugins.context_engine.dcp.types import MessageAnalysis

logger = logging.getLogger(__name__)


def deduplicate_tool_results(
    messages: List[Dict[str, Any]],
    analyses: List[MessageAnalysis],
) -> Tuple[List[Dict[str, Any]], List[MessageAnalysis]]:
    """Remove duplicate tool results, keeping only the most recent."""
    seen_content: Dict[str, int] = {}
    to_remove: Set[int] = set()

    for analysis in analyses:
        if not analysis.is_tool_result or analysis.is_protected:
            continue
        text = extract_content_text(messages[analysis.index])
        if text in seen_content:
            to_remove.add(seen_content[text])
            seen_content[text] = analysis.index
        else:
            seen_content[text] = analysis.index

    if to_remove:
        logger.info("DCP deduplication: removing %d duplicate tool results", len(to_remove))

    new_messages = [m for i, m in enumerate(messages) if i not in to_remove]
    new_analyses = [a for a in analyses if a.index not in to_remove]
    # Re-index
    for i, a in enumerate(new_analyses):
        a.index = i

    return new_messages, new_analyses


def purge_error_chains(
    messages: List[Dict[str, Any]],
    analyses: List[MessageAnalysis],
) -> Tuple[List[Dict[str, Any]], List[MessageAnalysis]]:
    """Remove tool call + result pairs where the result indicates an error."""
    to_remove: Set[int] = set()

    for i, analysis in enumerate(analyses):
        if not analysis.is_error or not is_tool_result(messages[analysis.index]):
            continue
        # Find the preceding assistant message with matching tool_call_id
        tool_call_id = messages[analysis.index].get("tool_call_id")
        if tool_call_id:
            for j in range(i - 1, -1, -1):
                msg = messages[j]
                if msg.get("role") != "assistant":
                    continue
                tool_calls = msg.get("tool_calls", [])
                if any(tc.get("id") == tool_call_id for tc in tool_calls):
                    to_remove.add(j)
                    to_remove.add(i)
                    break

    if to_remove:
        logger.info("DCP error purge: removing %d messages in error chains", len(to_remove))

    new_messages = [m for i, m in enumerate(messages) if i not in to_remove]
    new_analyses = [a for a in analyses if a.index not in to_remove]
    for i, a in enumerate(new_analyses):
        a.index = i

    return new_messages, new_analyses
