"""Field visualization (optional matplotlib dependency)."""
from __future__ import annotations
import numpy as np


def plot_field_summary(engine, title: str = "Field State", show: bool = True):
    """Plot V, knots, crystal, and stress fields for channel 0."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{title} (t={engine.t})")

    im0 = axes[0, 0].imshow(engine.V[:, :, 0], cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 0].set_title("V (void) ch0")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(engine.knots[:, :, 0], cmap="hot")
    axes[0, 1].set_title("Knots ch0")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(engine.crystal[:, :, 0], cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title("Crystal ch0")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(engine.stress, cmap="YlOrRd")
    axes[1, 1].set_title("Stress")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    if show:
        plt.show()
    return fig
