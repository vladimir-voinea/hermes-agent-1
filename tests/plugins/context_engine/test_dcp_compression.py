"""Tests for DCP compression."""

import pytest

from plugins.context_engine.dcp.compression import (
    compress_ranges,
    find_compressible_ranges,
    summarize_range,
)
from plugins.context_engine.dcp.message_utils import analyze_messages, extract_content_text


class TestFindCompressibleRanges:
    def test_finds_middle_range(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        analyses = analyze_messages(messages)
        ranges = find_compressible_ranges(messages, analyses)
        assert len(ranges) == 1
        assert ranges[0] == (3, 6)

    def test_no_ranges_when_all_protected(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        analyses = analyze_messages(messages)
        ranges = find_compressible_ranges(messages, analyses)
        assert ranges == []

    def test_multiple_ranges(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "q1"},  # protected (head)
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "[CONTEXT COMPACTION]"},  # protected
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        analyses = analyze_messages(messages, head_count=3)
        ranges = find_compressible_ranges(messages, analyses)
        # Should find range before and after protected compaction message
        assert len(ranges) == 2


class TestSummarizeRange:
    def test_summarizes_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        summary = summarize_range(messages, 0, 1)
        assert "[user]" in summary
        assert "[assistant]" in summary
        assert "hello" in summary
        assert "hi there" in summary

    def test_focus_topic(self):
        messages = [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
        ]
        summary = summarize_range(messages, 0, 1, focus_topic="math")
        assert "Focus: math" in summary


class TestCompressRanges:
    def test_compresses_range(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there, how can I help?"},
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
            {"role": "user", "content": "what is 3+3?"},
            {"role": "assistant", "content": "3+3 = 6"},
        ]
        analyses = analyze_messages(messages)
        new_msgs, saved = compress_ranges(messages, analyses)
        assert len(new_msgs) < len(messages)
        assert any("[DCP COMPRESSION" in extract_content_text(m) for m in new_msgs)

    def test_preserves_order(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        analyses = analyze_messages(messages)
        new_msgs, _ = compress_ranges(messages, analyses)
        assert new_msgs[0]["role"] == "system"
        assert new_msgs[1]["role"] == "user"

    def test_tokens_saved_positive(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi " * 100},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1 " * 100},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2 " * 100},
        ]
        analyses = analyze_messages(messages)
        _, saved = compress_ranges(messages, analyses)
        assert saved > 0
