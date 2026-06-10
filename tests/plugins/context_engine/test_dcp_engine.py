"""Integration tests for DCP context engine."""

import pytest

from plugins.context_engine.dcp.engine import DCPContextEngine


class TestDCPContextEngine:
    def test_name(self):
        engine = DCPContextEngine()
        assert engine.name == "dcp"

    def test_should_compress_when_over_threshold(self):
        engine = DCPContextEngine()
        engine.context_length = 100000
        engine.threshold_tokens = 70000
        engine.last_prompt_tokens = 75000
        assert engine.should_compress() is True

    def test_should_not_compress_when_under_threshold(self):
        engine = DCPContextEngine()
        engine.context_length = 100000
        engine.threshold_tokens = 70000
        engine.last_prompt_tokens = 50000
        assert engine.should_compress() is False

    def test_compress_reduces_message_count(self):
        engine = DCPContextEngine()
        messages = [
            {"role": "system", "content": "You are Hermes"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "3+3 = 6"},
        ]
        result = engine.compress(messages)
        assert len(result) < len(messages)

    def test_compress_preserves_system_and_head(self):
        engine = DCPContextEngine()
        engine.protect_first_n = 2
        messages = [
            {"role": "system", "content": "You are Hermes"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        result = engine.compress(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_on_session_start_clears_counters(self):
        engine = DCPContextEngine()
        engine.compression_count = 5
        engine.on_session_start("test-session")
        assert engine.compression_count == 0
        assert engine._session_id == "test-session"

    def test_get_tool_schemas(self):
        engine = DCPContextEngine()
        schemas = engine.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "dcp_status" in names
        assert "dcp_compress" in names

    def test_handle_tool_call_status(self):
        engine = DCPContextEngine()
        result = engine.handle_tool_call("dcp_status", {})
        import json
        data = json.loads(result)
        assert data["engine"] == "dcp"

    def test_handle_tool_call_compress(self):
        engine = DCPContextEngine()
        result = engine.handle_tool_call("dcp_compress", {"focus_topic": "math"})
        import json
        data = json.loads(result)
        assert data["triggered"] is True
        assert data["focus_topic"] == "math"

    def test_has_content_to_compress_when_enough_messages(self):
        engine = DCPContextEngine()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        assert engine.has_content_to_compress(messages) is True

    def test_has_content_to_compress_when_too_few(self):
        engine = DCPContextEngine()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert engine.has_content_to_compress(messages) is False

    def test_update_from_response(self):
        engine = DCPContextEngine()
        engine.update_from_response({
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        })
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 500
        assert engine.last_total_tokens == 1500

    def test_should_compress_preflight(self):
        engine = DCPContextEngine()
        engine.context_length = 100000
        engine.threshold_tokens = 70000
        messages = [{"role": "user", "content": "a" * 300000}]  # ~75k tokens
        assert engine.should_compress_preflight(messages) is True

    def test_get_status(self):
        engine = DCPContextEngine()
        engine.context_length = 100000
        engine.last_prompt_tokens = 50000
        status = engine.get_status()
        assert status["engine"] == "dcp"
        assert status["usage_percent"] == 50.0
