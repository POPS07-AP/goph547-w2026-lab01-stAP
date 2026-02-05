import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

TOTAL_MASS = 1e7
xm_target = np.array([0.0, 0.0, -10.0])

def generate_mass_set():
    masses = np.random.normal(TOTAL_MASS/5, TOTAL_MASS/100, 4)
    coords = np.column_stack([
        np.random.normal(0, 20, 4),
        np.random.normal(0, 20, 4),
        np.random.normal(-10, 2, 4)
    ])

    m5 = TOTAL_MASS - np.sum(masses)
    x5 = (TOTAL_MASS*xm_target - np.sum(masses[:,None]*coords, axis=0)) / m5

    masses = np.append(masses, m5)
    coords = np.vstack([coords, x5])

    return masses, coords

def compute_total_fields(X, Y, z, masses, coords):
    U = np.zeros_like(X)
    gz = np.zeros_like(X)

    for m, xm in zip(masses, coords):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i,j], Y[i,j], z])
                U[i,j] += gravity_potential_point(x, xm, m)
                gz[i,j] += gravity_effect_point(x, xm, m)
    return U, gz

for idx in range(1, 4):
    masses, coords = generate_mass_set()
    sio.savemat(f"mass_set_{idx}.mat", {"masses": masses, "coords": coords})
