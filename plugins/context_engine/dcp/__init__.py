"""DCP (Dynamic Context Pruning) context engine plugin."""

from plugins.context_engine.dcp.engine import DCPContextEngine


def register(ctx):
    """Register the DCP context engine and slash command."""
    engine = DCPContextEngine()
    ctx.register_context_engine(engine)

    ctx.register_command(
        "dcp",
        _handle_dcp_command,
        description="Trigger Dynamic Context Pruning manually",
        args_hint="[focus_topic]",
    )


def _handle_dcp_command(cmd_text: str) -> str:
    """Handle /dcp slash command."""
    parts = cmd_text.strip().split(None, 1)
    focus = parts[1] if len(parts) > 1 else None
    return f"DCP compression triggered{f' with focus: {focus}' if focus else ''}"
