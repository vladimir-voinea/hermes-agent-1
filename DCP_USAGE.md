# Dynamic Context Pruning (DCP) — Usage Guide

## What is DCP?

Dynamic Context Pruning (DCP) is a context engine plugin for Hermes Agent that
intelligently reduces conversation token usage using fast, deterministic,
rule-based strategies — no auxiliary LLM call required.

## Strategies

1. **Tool output deduplication** — Identical tool results across turns are
   reduced to the most recent occurrence.
2. **Error chain purge** — Failed tool call chains (assistant tool call +
   error result) are removed entirely.
3. **Range compression** — Contiguous blocks of old messages are summarized
   into a compact `[DCP COMPRESSION]` reference message.
4. **Protected content detection** — System prompts, head messages, and active
   task markers are never pruned.

## Activation

Add to `~/.hermes/config.yaml`:

```yaml
context:
  engine: dcp
```

To switch back to the built-in compressor:

```yaml
context:
  engine: compressor
```

## Manual control

During a Hermes session, trigger compression manually:

```
/dcp
/dcp math          # compress with focus on "math" topic
```

## Running Hermes from this branch

```bash
cd /Users/vladimir/dev/hermes-agent/dcp-branch
python3 -m hermes_cli.main
```

Or if Hermes is installed in your environment:

```bash
hermes
```

## Running DCP tests

```bash
cd /Users/vladimir/dev/hermes-agent/dcp-branch
python3 -m pytest tests/plugins/context_engine/ -v
```

## Files added by this plugin

- `plugins/context_engine/dcp/__init__.py`
- `plugins/context_engine/dcp/plugin.yaml`
- `plugins/context_engine/dcp/types.py`
- `plugins/context_engine/dcp/message_utils.py`
- `plugins/context_engine/dcp/strategies.py`
- `plugins/context_engine/dcp/compression.py`
- `plugins/context_engine/dcp/engine.py`
- `tests/plugins/context_engine/test_dcp_strategies.py`
- `tests/plugins/context_engine/test_dcp_compression.py`
- `tests/plugins/context_engine/test_dcp_engine.py`
