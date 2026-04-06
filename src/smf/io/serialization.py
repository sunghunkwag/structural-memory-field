"""State serialization with config preservation."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from smf.config.params import EngineConfig


def save_state(path: str | Path, state: dict) -> None:
    """Save engine state to .npz file (includes config)."""
    np.savez(str(path), **{k: v for k, v in state.items()
                           if isinstance(v, np.ndarray)},
             _meta=np.array(str({k: v for k, v in state.items()
                                  if not isinstance(v, np.ndarray)})))


def load_state(path: str | Path) -> dict:
    """Load engine state from .npz file."""
    data = np.load(str(path), allow_pickle=False)
    state = {}
    for k in data.files:
        if k == "_meta":
            import ast
            meta = ast.literal_eval(str(data[k]))
            state.update(meta)
        else:
            state[k] = data[k]
    return state
