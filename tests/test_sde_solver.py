import numpy as np

from src.sde_solver import SDEConfig, euler_maruyama_step, run_sde_simulation, theoretical_pdf


def test_euler_maruyama_step_deterministic_seed():
    cfg = SDEConfig(alpha=0.5, u=1.0, sigma=0.1, dt=0.01, steps=10, n_trajectories=2, seed=42)
    rng = np.random.default_rng(cfg.seed)
    q0 = np.zeros(2)
    q1 = euler_maruyama_step(q0, cfg.alpha, cfg.u, cfg.sigma, cfg.dt, rng)
    rng2 = np.random.default_rng(cfg.seed)
    q1_again = euler_maruyama_step(q0, cfg.alpha, cfg.u, cfg.sigma, cfg.dt, rng2)
    np.testing.assert_allclose(q1, q1_again)


def test_run_sde_simulation_shapes():
    cfg = SDEConfig(alpha=0.2, u=1.0, sigma=0.1, dt=0.01, steps=100, n_trajectories=3, seed=1)
    t, traj = run_sde_simulation(cfg, q0=0.0, record_interval=10)
    assert traj.shape[1] == cfg.n_trajectories
    assert traj.shape[0] == len(t)


def test_theoretical_pdf_normalization():
    q = np.linspace(-3, 3, 301)
    pdf = theoretical_pdf(q, alpha=0.5, u=1.0, sigma=0.3)
    integral = np.trapezoid(pdf, q)
    assert np.isclose(integral, 1.0, rtol=1e-3)
