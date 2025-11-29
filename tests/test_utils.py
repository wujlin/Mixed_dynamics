import numpy as np

from src.utils import calculate_autocorrelation


def test_calculate_autocorrelation_basic():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    r1 = calculate_autocorrelation(x, lag=1)
    assert 0 < r1 < 1


def test_calculate_autocorrelation_zero_variance():
    x = np.ones(10)
    r = calculate_autocorrelation(x, lag=1)
    assert r == 0.0
