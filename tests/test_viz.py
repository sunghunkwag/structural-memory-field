"""Viz tests — verify plot uses actual field data, not zeros."""
import numpy as np
import pytest
from smf.engine.v2 import EngineV2
from smf.viz.field_plots import plot_field_state


def test_viz_accepts_field_state():
    """plot_field_state takes FieldState, not engine."""
    import matplotlib
    matplotlib.use("Agg")

    e = EngineV2(knot_threshold=0.3)
    for _ in range(50):
        e.step(np.array([2, 0, 0, 0, 0, 0, 0, 0]))

    fig = plot_field_state(e.f, show=False)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_viz_plots_real_data():
    """Verify plot uses actual field data, not zeros."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from unittest.mock import patch, MagicMock

    e = EngineV2(knot_threshold=0.3)
    for _ in range(50):
        e.step(np.array([2, 0, 0, 0, 0, 0, 0, 0]))

    plotted = []

    def capture_imshow(data, **kwargs):
        plotted.append(np.array(data))
        # Return a real ScalarMappable so colorbar doesn't crash
        sm = plt.cm.ScalarMappable()
        sm.set_array(np.array(data))
        return sm

    with patch("matplotlib.axes.Axes.imshow", side_effect=capture_imshow):
        plot_field_state(e.f, show=False)

    assert len(plotted) >= 4  # at least 4 subplots
    assert any(np.std(d) > 0.001 for d in plotted)  # real data, not zeros

    plt.close("all")
