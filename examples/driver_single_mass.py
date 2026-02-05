"""
Driver script for single mass anomaly analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def create_plots(grid_spacing=5.0):
    """
    Create contour plots for single mass anomaly.

    Parameters
    ----------
    grid_spacing : float
        Grid spacing in meters.
    """

    # Mass parameters
    m = 1.0e7  # 10 million kg
    xm = np.array([0.0, 0.0, -10000.0])  # Centroid at 10km depth

    # Grid parameters
    x_min, x_max = -100.0, 100.0
    y_min, y_max = -100.0, 100.0

    # Create grid
    x = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X, Y = np.meshgrid(x, y)

    # Elevations to plot
    elevations = [0.0, 10.0, 100.0]

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f'Single Mass Anomaly (m={m:.1e} kg, grid spacing={grid_spacing}m)',
                 fontsize=14, y=0.98)

    # Initialize min/max for consistent colorbars
    U_min, U_max = float('inf'), float('-inf')
    gz_min, gz_max = float('inf'), float('-inf')

    # Pre-calculate to get consistent colorbar limits
    U_data = {}
    gz_data = {}

    for z in elevations:
        U_grid = np.zeros_like(X)
        gz_grid = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                survey_point = [X[i, j], Y[i, j], z]
                U_grid[i, j] = gravity_potential_point(survey_point, xm, m)
                gz_grid[i, j] = gravity_effect_point(survey_point, xm, m)

        U_data[z] = U_grid
        gz_data[z] = gz_grid

        U_min = min(U_min, U_grid.min())
        U_max = max(U_max, U_grid.max())
        gz_min = min(gz_min, gz_grid.min())
        gz_max = max(gz_max, gz_grid.max())

    # Create plots
    for idx, z in enumerate(elevations):
        # Potential plot
        ax1 = axes[idx, 0]
        contour1 = ax1.contourf(X, Y, U_data[z], 50, cmap='viridis',
                                vmin=U_min, vmax=U_max)
        ax1.scatter(X.flatten(), Y.flatten(), marker='x', color='black',
                    s=2, alpha=0.5, label='Grid points')
        ax1.set_title(f'Gravity Potential at z={z}m')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Add colorbar to first plot only
        if idx == 0:
            plt.colorbar(contour1, ax=ax1, label='Potential [m²/s²]')

        # Gravity effect plot
        ax2 = axes[idx, 1]
        contour2 = ax2.contourf(X, Y, gz_data[z], 50, cmap='plasma',
                                vmin=gz_min, vmax=gz_max)
        ax2.scatter(X.flatten(), Y.flatten(), marker='x', color='black',
                    s=2, alpha=0.5, label='Grid points')
        ax2.set_title(f'Gravity Effect at z={z}m')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Add colorbar to first plot only
        if idx == 0:
            plt.colorbar(contour2, ax=ax2, label='g_z [m/s²]')

    plt.tight_layout()

    # Save figure
    filename = f'single_mass_grid_{int(grid_spacing)}m.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as {filename}")

    plt.show()


def main():
    """Main function."""
    print("Generating plots for single mass anomaly...")

    # Create plots for both grid spacings
    for spacing in [5.0, 25.0]:
        create_plots(grid_spacing=spacing)

    print("Analysis complete!")


if __name__ == "__main__":
    main()