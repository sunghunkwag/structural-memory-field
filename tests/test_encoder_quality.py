"""Encoder quality tests — verify TextEncoder produces real, meaningful vectors."""
import numpy as np
from smf.io.encoders import TextEncoder


def test_encoder_different_outputs():
    enc = TextEncoder(K=8)
    a1, p1 = enc.encode("cat")
    a2, p2 = enc.encode("dog")
    assert not np.allclose(a1, a2)
    assert not np.allclose(p1, p2)


def test_encoder_similarity_preserved():
    enc = TextEncoder(K=8)
    a_cat, _ = enc.encode("cat")
    a_cats, _ = enc.encode("cats")
    a_xyz, _ = enc.encode("xylophone")
    assert np.linalg.norm(a_cat - a_cats) < np.linalg.norm(a_cat - a_xyz)


def test_encoder_multi_channel():
    enc = TextEncoder(K=8)
    a, p = enc.encode("hello world")
    assert np.sum(np.abs(a) > 0.01) >= 3  # at least 3 active channels


def test_encoder_phase_nontrivial():
    enc = TextEncoder(K=8)
    _, p1 = enc.encode("alpha")
    _, p2 = enc.encode("beta")
    assert not np.allclose(p1, 0.0)
    assert not np.allclose(p1, p2)


def test_encoder_deterministic():
    enc = TextEncoder(K=8)
    a1, p1 = enc.encode("test")
    a2, p2 = enc.encode("test")
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(p1, p2)
