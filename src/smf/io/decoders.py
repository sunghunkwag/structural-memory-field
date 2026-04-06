"""Output decoders — interpret engine action/state."""
from __future__ import annotations


def decode_action(action: int, labels: list[str] | None = None) -> str:
    """Convert action index to label."""
    if labels and action < len(labels):
        return labels[action]
    return f"action_{action}"
