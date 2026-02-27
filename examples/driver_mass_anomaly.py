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
    # check if mass anomaly data file is present
    if not os.path.exists("anomaly_data.mat"):
        raise FileNotFoundError("Missing file anomaly_data.mat")

    # load data set
    data = loadmat("anomaly_data.mat")
    rho = data["rho"]
    xm = data["x"]
    ym = data["y"]
    zm = data["z"]

    # compute mass anomaly statistics
    vcell = 2.0**3
    mm = rho * vcell
    vtot = 200.0 * 200.0 * 30.0
    mtot = np.sum(mm.flatten())
    xbar = np.sum((xm * mm).flatten()) / mtot
    ybar = np.sum((ym * mm).flatten()) / mtot
    zbar = np.sum((zm * mm).flatten()) / mtot
    rho_max = np.max(rho.flatten())
    rho_bar = np.mean(rho.flatten())

    print()
    print(f"vtot = {vtot:.3e} m^3")
    print(f"mtot = {mtot:.3e} kg")
    print(f"xbar = [{xbar:.3f}, {ybar:.3f}, {zbar:.3f}]")
    print(f"rho_max = {rho_max:.3e} kg/m^3")
    print(f"rho_bar = {rho_bar:.3e} kg/m^3")
    print()

    # specify region where density is non-negligible
    kx_min = 40
    kx_max = 60
    xmin = xm[0, kx_min, 0]
    xmax = xm[0, kx_max, 0]
    ky_min = 44
    ky_max = 56
    ymin = ym[ky_min, 0, 0]
    ymax = ym[ky_max, 0, 0]
    kz_min = 7
    kz_max = 13
    zmin = zm[0, 0, kz_min]
    zmax = zm[0, 0, kz_max]
    print("xmin, xmax: [", xmin, xmax, "] m")
    print("ymin, ymax: [", ymin, ymax, "] m")
    print("zmin, zmax: [", zmin, zmax, "] m")
    vtot_sub = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    mtot_sub = np.sum(mm[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1])
    rho_bar_sub = np.mean(
        rho[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1].flatten()
    )
    print(f"vtot_sub = {vtot_sub:.3e} m^3")
    print(f"mtot_sub = {mtot_sub:.3e} kg")
    print(f"rho_bar_sub = {rho_bar_sub:.3e} kg/m^3")

    # extract subregion of density data for survey later
    mm_sub = mm[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1].flatten()
    xm_sub = xm[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1].flatten()
    ym_sub = ym[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1].flatten()
    zm_sub = zm[ky_min : ky_max + 1, kx_min : kx_max + 1, kz_min : kz_max + 1].flatten()

    # plot density distribution
    plt.figure(figsize=(8, 9))
    rho_bar_min = 0.0
    rho_bar_max = 0.6

    # mean density along y-axis
    plt.subplot(3, 1, 1)
    plt.contourf(
        xm[0, :, :],
        zm[0, :, :],
        np.mean(rho, axis=0),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(xbar, zbar, "xk", markersize=5)
    plt.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [zmin, zmax, zmax, zmin, zmin],
        "--k",
    )
    cbar = plt.colorbar(ticks=np.linspace(rho_bar_min, rho_bar_max, 7))
    cbar.set_label(r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Mean density along y-axis")
    plt.xlim((-30, 30))
    plt.ylim((-20, 0))

    # mean density along x-axis
    plt.subplot(3, 1, 2)
    plt.contourf(
        ym[:, 0, :],
        zm[:, 0, :],
        np.mean(rho, axis=1),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(ybar, zbar, "xk", markersize=5)
    plt.plot(
        [ymin, ymin, ymax, ymax, ymin],
        [zmin, zmax, zmax, zmin, zmin],
        "--k",
    )
    cbar = plt.colorbar(ticks=np.linspace(rho_bar_min, rho_bar_max, 7))
    cbar.set_label(r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.title("Mean density along x-axis")
    plt.xlim((-30, 30))
    plt.ylim((-20, 0))

    # mean density along z-axis
    plt.subplot(3, 1, 3)
    plt.contourf(
        xm[:, :, 0],
        ym[:, :, 0],
        np.mean(rho, axis=2),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(xbar, ybar, "xk", markersize=5)
    plt.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [ymin, ymax, ymax, ymin, ymin],
        "--k",
    )
    cbar = plt.colorbar(ticks=np.linspace(rho_bar_min, rho_bar_max, 7))
    cbar.set_label(r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Mean density along z-axis")
    plt.xlim((-30, 30))
    plt.ylim((-30, 30))

    plt.subplots_adjust(hspace=0.5)

    plt.savefig("anomaly_mean_density.png", dpi=300)
    plt.close()

    # generate and load survey data
    if not os.path.exists("anomaly_survey_data.mat"):
        print()
        print("Survey data file not found, running survey: ")
        generate_survey_data(mm_sub, xm_sub, ym_sub, zm_sub)

    survey_data = loadmat("anomaly_survey_data.mat")
    x_5 = survey_data["x_5"]
    y_5 = survey_data["y_5"]
    zp = survey_data["zp"][0]
    g_5 = survey_data["g_5"]
    U_5 = survey_data["U_5"]
    dx = x_5[0, 1] - x_5[0, 0]
    dy = y_5[1, 0] - y_5[0, 0]
    print()
    print("Survey data loaded!")

    # compute derivatives
    dgdz = np.stack(
        (
            (g_5[:, :, 1] - g_5[:, :, 0]) / (zp[1] - zp[0]),
            (g_5[:, :, 3] - g_5[:, :, 2]) / (zp[3] - zp[2]),
        ),
        axis=-1,
    )
    # d^2g / dz^2 = -(d^2g/dx^2 + d^2g/dy^2) from Laplace equation
    d2gdz2 = np.stack(
        (
            # z = 0.0 m
            -(g_5[2:, 1:-1, 0] - 2 * g_5[1:-1, 1:-1, 0] + g_5[:-2, 1:-1, 0]) / dy**2
            - (g_5[1:-1, 2:, 0] - 2 * g_5[1:-1, 1:-1, 0] + g_5[1:-1, :-2, 0]) / dx**2,
            # z = 100.0 m
            -(g_5[2:, 1:-1, 2] - 2 * g_5[1:-1, 1:-1, 2] + g_5[:-2, 1:-1, 2]) / dy**2
            - (g_5[1:-1, 2:, 2] - 2 * g_5[1:-1, 1:-1, 2] + g_5[1:-1, :-2, 2]) / dx**2,
        ),
        axis=-1,
    )

    # plot vertical gravity effect
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.contourf(
        x_5,
        y_5,
        g_5[:, :, 0],
        cmap="viridis_r",
        levels=np.linspace(0.0, 1.2e-9, 50),
    )
    plt.ylabel("y [m]")
    cbar = plt.colorbar(ticks=np.linspace(0.0, 1.2e-9, 7))
    cbar.set_label(r"$g_z$ [$m/s^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(2, 2, 2)
    plt.contourf(
        x_5,
        y_5,
        g_5[:, :, 1],
        cmap="viridis_r",
        levels=np.linspace(0.0, 1.2e-9, 50),
    )
    cbar = plt.colorbar(ticks=np.linspace(0.0, 1.2e-9, 7))
    cbar.set_label(r"$g_z$ [$m/s^2$]")
    plt.text(-90, 70, "z = 1.0 m", weight="bold", bbox=dict(facecolor="white"))

    plt.subplot(2, 2, 3)
    plt.contourf(
        x_5,
        y_5,
        g_5[:, :, 2],
        cmap="viridis_r",
        levels=np.linspace(0.0, 2.0e-11, 50),
    )
    cbar = plt.colorbar(ticks=np.linspace(0.0, 2.0e-11, 6))
    cbar.set_label(r"$g_z$ [$m/s^2$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.subplot(2, 2, 4)
    plt.contourf(
        x_5,
        y_5,
        g_5[:, :, 3],
        cmap="viridis_r",
        levels=np.linspace(0.0, 2.0e-11, 50),
    )
    cbar = plt.colorbar(ticks=np.linspace(0.0, 2.0e-11, 6))
    cbar.set_label(r"$g_z$ [$m/s^2$]")
    plt.text(-90, 70, "z = 110.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlabel("x [m]")

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig("anomaly_survey_data.png", dpi=300)
    plt.close()

    # plot first and second derivatives of gravity effect
    plt.figure(figsize=(8, 8))

    # dg/dz at z = 0 m
    plt.subplot(2, 2, 1)
    plt.contourf(
        x_5,
        y_5,
        dgdz[:, :, 0],
        cmap="viridis",
        levels=np.linspace(-1.4e-10, 0.2e-10, 50),
    )
    plt.ylabel("y [m]")
    cbar = plt.colorbar(ticks=np.linspace(-1.4e-10, 0.2e-10, 9))
    cbar.set_label(r"$\partial g_z / \partial z$ [$(m/s^2) / m$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))

    # dg/dz at z = 100 m
    plt.subplot(2, 2, 3)
    plt.contourf(
        x_5,
        y_5,
        dgdz[:, :, 1],
        cmap="viridis",
        levels=np.linspace(-2.8e-13, 0.0, 50),
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(ticks=np.linspace(-2.8e-13, 0.0, 8))
    cbar.set_label(r"$\partial g_z / \partial z$ [$(m/s^2) / m$]")
    plt.text(-90, 70, "z = 100.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))

    # d2g/dz2 at z = 0 m
    plt.subplot(2, 2, 2)
    plt.contourf(
        x_5[1:-1, 1:-1],
        y_5[1:-1, 1:-1],
        d2gdz2[:, :, 0],
        cmap="viridis_r",
        levels=np.linspace(-0.4e-11, 2.4e-11, 50),
    )
    cbar = plt.colorbar(ticks=np.linspace(-0.4e-11, 2.4e-11, 8))
    cbar.set_label(r"$\partial^2 g_z / \partial z^2$ [$(m/s^2) / m^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))

    # d2g/dz2 at z = 100 m
    plt.subplot(2, 2, 4)
    plt.contourf(
        x_5[1:-1, 1:-1],
        y_5[1:-1, 1:-1],
        d2gdz2[:, :, 1],
        cmap="viridis_r",
        levels=np.linspace(-1.0e-15, 8.0e-15, 50),
    )
    plt.xlabel("x [m]")
    cbar = plt.colorbar(ticks=np.linspace(-1.0e-15, 8e-15, 10))
    cbar.set_label(r"$\partial^2 g_z / \partial z^2$ [$(m/s^2) / m^2$]")
    plt.text(-90, 70, "z = 0.0 m", weight="bold", bbox=dict(facecolor="white"))
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.savefig("anomaly_survey_derivatives.png", dpi=300)
    plt.close()


def generate_survey_data(mm_sub, xm_sub, ym_sub, zm_sub):
    # simulate vertical gravity effect survey at z = {0, 1, 100, 110} m
    # and grid spacing of dx = dy = 5 m

    # survey grids
    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41), np.linspace(-100.0, 100.0, 41)
    )
    zp = [0.0, 1.0, 100.0, 110.0]

    # generate survey data
    U_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    xs = x_5[0, :]
    ys = y_5[:, 0]
    Us = U_5
    gs = g_5
    for km, (mm_k, xx_k, yy_k, zz_k) in enumerate(zip(mm_sub, xm_sub, ym_sub, zm_sub)):
        xm_k = [xx_k, yy_k, zz_k]
        for k, zz in enumerate(zp):
            for j, xx in enumerate(xs):
                for i, yy in enumerate(ys):
                    x = [xx, yy, zz]
                    Us[i, j, k] += gravity_potential_point(x, xm_k, mm_k)
                    gs[i, j, k] += gravity_effect_point(x, xm_k, mm_k)

    # save as a .mat file so this only needs to be calculated once
    savemat(
        "anomaly_survey_data.mat",
        mdict={
            "x_5": x_5,
            "y_5": y_5,
            "zp": zp,
            "g_5": g_5,
            "U_5": U_5,
        },
    )
    print("Survey data generated!")


if __name__ == "__main__":
    main()
