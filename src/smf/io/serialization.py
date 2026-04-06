"""State serialization with config preservation."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def save_state(engine, path: str | Path) -> None:
    """Save engine state + config to .npz file.

    Args:
        engine: EngineV2 or EngineV3 instance.
        path: File path (should end in .npz).
    """
    state = engine.get_state()
    arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    arrays["_meta_json"] = np.array(json.dumps(meta, default=_json_default))
    np.savez(str(path), **arrays)


def load_state(path: str | Path) -> dict:
    """Load state dict from .npz file.

    Returns dict compatible with engine.load_state().
    """
    data = np.load(str(path), allow_pickle=False)
    state = {}
    for k in data.files:
        if k == "_meta_json":
            meta = json.loads(str(data[k]))
            state.update(meta)
        elif k == "_meta":
            # Legacy format from previous serialization
            import ast
            meta = ast.literal_eval(str(data[k]))
            state.update(meta)
        else:
            state[k] = data[k]
    return state


def _json_default(obj):
    """JSON serializer for types not natively serializable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
