import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (
    savemat,
    loadmat,
)

from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)


def main():
    # check if mass anomalies have already been generated
    # if not, generate them
    if (
        not os.path.exists("mass_set_0.mat")
        or not os.path.exists("mass_set_1.mat")
        or not os.path.exists("mass_set_2.mat")
    ):
        generate_mass_anomaly_sets()

    # survey grids
    x_25, y_25 = np.meshgrid(
        np.linspace(-100.0, 100.0, 9), np.linspace(-100.0, 100.0, 9)
    )
    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41), np.linspace(-100.0, 100.0, 41)
    )
    zp = [0.0, 10.0, 100.0]

    # load data set
    for kk in range(3):
        data = loadmat(f"mass_set_{kk}.mat")
        m = data["m"][:, 0]
        xm = data["xm"]
        print()
        print(f"Mass set {kk}")
        print()
        print(f"m:\n{m}")
        print(f"mtot = {np.sum(m):.2e}")
        print()
        print(f"xm:\n{xm}")
        print(f"xbar: {np.dot(m.T, xm) / np.sum(m)}")
        print()

        # survey at 25 m grid spacing
        U_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
        g_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
        xs = x_25[0, :]
        ys = y_25[:, 0]
        Us = U_25
        gs = g_25
        for km, mm in enumerate(m):
            for k, zz in enumerate(zp):
                for j, xx in enumerate(xs):
                    for i, yy in enumerate(ys):
                        x = [xx, yy, zz]
                        Us[i, j, k] += gravity_potential_point(x, xm[km, :], mm)
                        gs[i, j, k] += gravity_effect_point(x, xm[km, :], mm)

        # survey at 5 m grid spacing
        U_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
        g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
        xs = x_5[0, :]
        ys = y_5[:, 0]
        Us = U_5
        gs = g_5
        for km, mm in enumerate(m):
            for k, zz in enumerate(zp):
                for j, xx in enumerate(xs):
                    for i, yy in enumerate(ys):
                        x = [xx, yy, zz]
                        Us[i, j, k] += gravity_potential_point(x, xm[km, :], mm)
                        gs[i, j, k] += gravity_effect_point(x, xm[km, :], mm)

        plot_gravity_potential_and_effect(
            kk,
            x_25,
            y_25,
            U_25,
            g_25,
            x_5,
            y_5,
            U_5,
            g_5,
        )


def generate_mass_anomaly_sets():
    # properties of mass anomaly set
    mtot = 1.0e7  # mass, in kg
    mbar = 0.2 * mtot
    msig = 0.01 * mtot

    xbar = np.array((0.0, 0.0, -10.0))  # position of mass, in m
    xsig = np.array((20.0, 20.0, 2.0))

    # generate mass anomaly sets
    for k in range(3):
        print()
        print(f"Mass set {k}")
        mm = np.random.normal(loc=mbar, scale=msig, size=(5, 1))
        mm[-1] = mtot - np.sum(mm[:-1])

        print(f"masses:\n{mm}")
        print(f"mtot = {np.sum(mm):.2e}")
        print()

        xx = np.zeros((5, 3))
        xx[:-1, 0] = np.random.normal(loc=xbar[0], scale=xsig[0], size=(4,))
        xx[-1, 0] = (xbar[0] * mtot - np.dot(mm[:-1, 0], xx[:-1, 0])) / mm[-1, 0]
        xx[:-1, 1] = np.random.normal(loc=xbar[1], scale=xsig[1], size=(4,))
        xx[-1, 1] = (xbar[1] * mtot - np.dot(mm[:-1, 0], xx[:-1, 1])) / mm[-1, 0]
        xx[:-1, 2] = np.random.normal(loc=xbar[2], scale=xsig[2], size=(4,))
        xx[-1, 2] = (xbar[2] * mtot - np.dot(mm[:-1, 0], xx[:-1, 2])) / mm[-1, 0]

        print(f"xm:\n{xx}")
        print(f"xbar: {np.dot(mm.T, xx) / mtot}")
        print(np.dot(xx[:, 0], mm[:, 0]) / mtot)
        print(np.dot(xx[:, 1], mm[:, 0]) / mtot)
        print(np.dot(xx[:, 2], mm[:, 0]) / mtot)
        print()

        savemat(f"mass_set_{k}.mat", mdict={"m": mm, "xm": xx})


def plot_gravity_potential_and_effect(
    kk,
    x_25,
    y_25,
    U_25,
    g_25,
    x_5,
    y_5,
    U_5,
    g_5,
):
    # generate plots of gravity potential and gravity effect, grid 25.0 m
    fig = plt.figure(figsize=(8, 8))
    Umin = 0.0
    Umax = 8.0e-5  # for colorbar limits
    gmin = 0.0
    gmax = 7.0e-6

    fig.suptitle(
        f"Mass Set {kk}, mtot = 1.0e7 kg, zm_bar = -10 m, xy_grid = 25.0 m",
        weight="bold",
    )

    # gravity potential, U
    plt.subplot(3, 2, 1)
    plt.contourf(
        x_25, y_25, U_25[:, :, 0], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 3)
    plt.contourf(
        x_25, y_25, U_25[:, :, 1], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 10.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 5)
    plt.contourf(
        x_25, y_25, U_25[:, :, 2], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))

    # gravity effect, gz
    plt.subplot(3, 2, 2)
    plt.contourf(
        x_25, y_25, g_25[:, :, 0], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 4)
    plt.contourf(
        x_25, y_25, g_25[:, :, 1], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 10.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 6)
    plt.contourf(
        x_25, y_25, g_25[:, :, 2], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    plt.plot(x_25, y_25, "xk", markersize=2)
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlabel("x [m]")

    plt.savefig(f"multi_mass_grid_25_set_{kk}.png", dpi=300)

    # generate plots of gravity potential and gravity effect, grid 5.0 m
    fig = plt.figure(figsize=(8, 8))
    Umin = 0.0
    Umax = 8.0e-5  # for colorbar limits
    gmin = 0.0
    gmax = 7.0e-6

    fig.suptitle(
        f"Mass Set {kk}, mtot = 1.0e7 kg, zm_bar = -10 m, xy_grid = 5.0 m",
        weight="bold",
    )

    # gravity potential, U
    plt.subplot(3, 2, 1)
    plt.contourf(
        x_5, y_5, U_5[:, :, 0], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 3)
    plt.contourf(
        x_5, y_5, U_5[:, :, 1], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 10.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 5)
    plt.contourf(
        x_5, y_5, U_5[:, :, 2], cmap="viridis_r", levels=np.linspace(Umin, Umax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(Umin, Umax, 5))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar.set_label(r"U [$m^2/s^2$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))

    # gravity effect, gz
    plt.subplot(3, 2, 2)
    plt.contourf(
        x_5, y_5, g_5[:, :, 0], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 4)
    plt.contourf(
        x_5, y_5, g_5[:, :, 1], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 10.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(3, 2, 6)
    plt.contourf(
        x_5, y_5, g_5[:, :, 2], cmap="viridis_r", levels=np.linspace(gmin, gmax, 500)
    )
    cbar = plt.colorbar(ticks=np.linspace(gmin, gmax, 5))
    cbar.set_label(r"g [$m/s^2$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlabel("x [m]")

    plt.savefig(f"multi_mass_grid_5_set_{kk}.png", dpi=300)


if __name__ == "__main__":
    main()
