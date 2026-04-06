"""Field visualization (optional matplotlib dependency).

Operates on FieldState objects, not engine instances — keeps viz decoupled.
"""
from __future__ import annotations
from smf.core.fields import FieldState


def plot_field_state(
    fields: FieldState,
    title: str = "Field State",
    channel: int = 0,
    show: bool = True,
):
    """Plot V, knots, crystal, and stress fields for a given channel.

    Args:
        fields: FieldState containing the field arrays.
        title: Plot title.
        channel: Which channel to visualize (default 0).
        show: Whether to call plt.show().

    Returns:
        matplotlib Figure, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{title} (t={fields.t})")

    im0 = axes[0, 0].imshow(fields.V[:, :, channel], cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 0].set_title(f"V (void) ch{channel}")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(fields.knots[:, :, channel], cmap="hot")
    axes[0, 1].set_title(f"Knots ch{channel}")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(fields.crystal[:, :, channel], cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Crystal ch{channel}")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(fields.stress, cmap="YlOrRd")
    axes[1, 1].set_title("Stress")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    if show:
        plt.show()
    return fig
