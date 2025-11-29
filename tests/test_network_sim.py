import numpy as np

from src.network_sim import NetworkAgentModel, NetworkConfig


def test_run_shapes_and_bounds():
    cfg = NetworkConfig(n=30, avg_degree=4, model="er", beta=0.0, seed=123)
    model = NetworkAgentModel(cfg)
    t, q, a = model.run(steps=20, record_interval=5)
    assert t.shape == q.shape == a.shape
    assert np.all(np.abs(q) <= 1.0)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_reproducibility_with_seed():
    cfg = NetworkConfig(n=20, avg_degree=3, model="ba", beta=0.1, seed=999)
    m1 = NetworkAgentModel(cfg)
    m2 = NetworkAgentModel(cfg)
    _, q1, a1 = m1.run(steps=15, record_interval=3)
    _, q2, a2 = m2.run(steps=15, record_interval=3)
    np.testing.assert_allclose(q1, q2)
    np.testing.assert_allclose(a1, a2)
