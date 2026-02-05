import numpy as np

def gravity_potential_point(x, xm, m, G=6.674e-11):
    """
    Compute the gravity potential due to a point mass.

    Parameters
    ----------
    x : array_like, shape (3,)
        Survey point coordinates [x, y, z]
    xm : array_like, shape (3,)
        Point mass coordinates
    m : float
        Mass of anomaly
    G : float
        Gravitational constant

    Returns
    -------
    float
        Gravity potential at x
    """
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)
    r = np.linalg.norm(x - xm)

    if r == 0:
        raise ValueError("Survey point cannot coincide with mass location.")

    return -G * m / r


def gravity_effect_point(x, xm, m, G=6.674e-11):
    """
    Compute vertical gravity effect (positive downward)
    """
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)

    dx = x - xm
    r = np.linalg.norm(dx)

    if r == 0:
        raise ValueError("Survey point cannot coincide with mass location.")

    gz = G * m * dx[2] / r**3   # vertical component
    return gz
