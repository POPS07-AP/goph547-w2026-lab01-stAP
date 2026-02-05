import numpy as np
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def test_potential_simple():
    x = np.array([0, 0, 0])
    xm = np.array([0, 0, -10])
    m = 1.0

    U = gravity_potential_point(x, xm, m)
    assert np.isclose(U, -6.674e-11 / 10)

def test_effect_simple():
    x = np.array([0, 0, 0])
    xm = np.array([0, 0, -10])
    m = 1.0

    gz = gravity_effect_point(x, xm, m)
    expected = 6.674e-11 * 10 / (10**3)
    assert np.isclose(gz, expected)
