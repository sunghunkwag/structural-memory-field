"""Shared fixtures for all tests."""
import numpy as np
import pytest
from smf.engine.v2 import EngineV2
from smf.engine.v3 import EngineV3
from smf.config.params import EngineConfig


@pytest.fixture
def s_A():
    return np.array([1, 0, 0, 0, 0, 0, 0, 0])

@pytest.fixture
def s_B():
    return np.array([0, 0, 1, 0, 0, 0, 0, 0])

@pytest.fixture
def empty():
    return np.zeros(8)

@pytest.fixture
def engine_v2():
    return EngineV2()

@pytest.fixture
def engine_v3():
    return EngineV3()
