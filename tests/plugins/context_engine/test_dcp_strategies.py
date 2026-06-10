"""Tests for DCP pruning strategies."""

import pytest

from plugins.context_engine.dcp.message_utils import analyze_messages
from plugins.context_engine.dcp.strategies import deduplicate_tool_results, purge_error_chains


def _make_tool_result(content: str, tool_call_id: str = "tc1"):
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def _make_assistant_with_tool(tool_call_id: str = "tc1", name: str = "test"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": name}}],
    }


class TestDeduplicateToolResults:
    def test_removes_duplicate_tool_results(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("same output", "tc1"),
            _make_assistant_with_tool("tc2"),
            _make_tool_result("same output", "tc2"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, new_analyses = deduplicate_tool_results(messages, analyses)
        assert len(new_msgs) == 5
        assert len([a for a in new_analyses if a.is_tool_result]) == 1

    def test_keeps_unique_results(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("output A", "tc1"),
            _make_assistant_with_tool("tc2"),
            _make_tool_result("output B", "tc2"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = deduplicate_tool_results(messages, analyses)
        assert len(new_msgs) == 6

    def test_keeps_most_recent_duplicate(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("same output", "tc1"),
            _make_assistant_with_tool("tc2"),
            _make_tool_result("same output", "tc2"),
            _make_assistant_with_tool("tc3"),
            _make_tool_result("same output", "tc3"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, new_analyses = deduplicate_tool_results(messages, analyses)
        # Should keep only the last duplicate
        tool_results = [a for a in new_analyses if a.is_tool_result]
        assert len(tool_results) == 1
        # The remaining one should be at the last position
        assert tool_results[0].index == 5  # re-indexed position (6 messages total)

    def test_no_tool_results_unchanged(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = deduplicate_tool_results(messages, analyses)
        assert len(new_msgs) == 3


class TestPurgeErrorChains:
    def test_removes_error_tool_call_chain(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("Error: file not found", "tc1"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = purge_error_chains(messages, analyses)
        assert len(new_msgs) == 2
        assert new_msgs[-1]["role"] == "user"

    def test_keeps_successful_calls(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("Success: file created", "tc1"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = purge_error_chains(messages, analyses)
        assert len(new_msgs) == 4

    def test_removes_multiple_error_chains(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("Error: failed", "tc1"),
            _make_assistant_with_tool("tc2"),
            _make_tool_result("Error: failed again", "tc2"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = purge_error_chains(messages, analyses)
        assert len(new_msgs) == 2

    def test_mixed_errors_and_success(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            _make_assistant_with_tool("tc1"),
            _make_tool_result("Success: created", "tc1"),
            _make_assistant_with_tool("tc2"),
            _make_tool_result("Error: failed", "tc2"),
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = purge_error_chains(messages, analyses)
        assert len(new_msgs) == 4  # keeps success, removes error chain
        assert any("Success" in str(m.get("content", "")) for m in new_msgs)
