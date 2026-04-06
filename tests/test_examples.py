"""Test that all examples run without errors and produce expected output."""
import subprocess
import sys


def _run_example(script):
    return subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=120,
    )


def test_basic_simulation_runs():
    r = _run_example("examples/basic_simulation.py")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"


def test_text_demo_runs():
    r = _run_example("examples/text_encoding_demo.py")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "Recall" in r.stdout


def test_pattern_demo_runs():
    r = _run_example("examples/pattern_memory_demo.py")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "unique actions" in r.stdout


def test_baseline_demo_runs():
    r = _run_example("examples/comparison_baseline.py")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "Random" in r.stdout
    assert "SMF" in r.stdout
