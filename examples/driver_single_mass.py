import numpy as np
import matplotlib.pyplot as plt

from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)


def main():
    # properties of mass anomaly
    m = 1.0e7  # mass, in kg
    xm = np.array((0.0, 0.0, -10.0))  # position of mass, in m

    # survey grids
    x_25, y_25 = np.meshgrid(
        np.linspace(-100.0, 100.0, 9), np.linspace(-100.0, 100.0, 9)
    )
    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41), np.linspace(-100.0, 100.0, 41)
    )
    zp = [0.0, 10.0, 100.0]

    # survey at 25 m grid spacing
    U_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    g_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    xs = x_25[0, :]
    ys = y_25[:, 0]
    Us = U_25
    gs = g_25
    for k, zz in enumerate(zp):
        for j, xx in enumerate(xs):
            for i, yy in enumerate(ys):
                x = [xx, yy, zz]
                Us[i, j, k] = gravity_potential_point(x, xm, m)
                gs[i, j, k] = gravity_effect_point(x, xm, m)

    # survey at 5 m grid spacing
    U_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    xs = x_5[0, :]
    ys = y_5[:, 0]
    Us = U_5
    gs = g_5
    for k, zz in enumerate(zp):
        for j, xx in enumerate(xs):
            for i, yy in enumerate(ys):
                x = [xx, yy, zz]
                Us[i, j, k] = gravity_potential_point(x, xm, m)
                gs[i, j, k] = gravity_effect_point(x, xm, m)

    # generate plots of gravity potential and gravity effect, grid 25.0 m
    fig = plt.figure(figsize=(8, 8))
    Umin = 0.0
    Umax = 8.0e-5  # for colorbar limits
    gmin = 0.0
    gmax = 7.0e-6

    fig.suptitle("m = 1.0e7 kg, zm = -10 m, xy_grid = 25.0 m", weight="bold")

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

    plt.savefig("single_mass_grid_25.png", dpi=300)

    # generate plots of gravity potential and gravity effect, grid 5.0 m
    fig = plt.figure(figsize=(8, 8))
    Umin = 0.0
    Umax = 8.0e-5  # for colorbar limits
    gmin = 0.0
    gmax = 7.0e-6

    fig.suptitle("m = 1.0e7 kg, zm = -10 m, xy_grid = 5.0 m", weight="bold")

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

    plt.savefig("single_mass_grid_5.png", dpi=300)


if __name__ == "__main__":
    main()
