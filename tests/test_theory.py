import numpy as np

from src import theory


def test_calculate_chi_symmetry_large_sensitivity():
    chi = theory.calculate_chi(phi=0.6, theta=0.4, k_avg=50)
    assert chi > 2.0


def test_calculate_rc_below_one_when_chi_large():
    chi = 4.0
    rc = theory.calculate_rc(n_m=10, n_w=5, chi=chi)
    assert rc < 1.0


def test_potential_energy_shape_and_values():
    q = np.array([-1.0, 0.0, 1.0])
    alpha, u = theory.get_gl_params(r=np.array([0.8, 1.0, 1.2]), rc=1.0)
    v = theory.potential_energy(q, alpha=alpha[1], u=u)
    expected = 0.25 * u * q**4  # alpha=0 对应单项 quartic
    np.testing.assert_allclose(v, expected)
