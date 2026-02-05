import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from goph547lab01.gravity import gravity_effect_point

# --------------------------------------------------
# Load anomaly data
# --------------------------------------------------
data = sio.loadmat("anomaly_data.mat")
x = data["x"]
y = data["y"]
z = data["z"]
rho = data["rho"]

print(f"Data shapes - x: {x.shape}, y: {y.shape}, z: {z.shape}, rho: {rho.shape}")

# --------------------------------------------------
# Convert density to mass & compute properties
# --------------------------------------------------
cell_volume = 2 * 2 * 2
masses = rho * cell_volume

total_mass = np.sum(masses)
x_bar = np.sum(x * masses) / total_mass
y_bar = np.sum(y * masses) / total_mass
z_bar = np.sum(z * masses) / total_mass

print("\n=== MASS PROPERTIES ===")
print(f"Total mass: {total_mass:.2e} kg")
print(f"Barycentre: ({x_bar:.1f}, {y_bar:.1f}, {z_bar:.1f}) m")
print(f"Max density: {np.max(rho):.1f} kg/m³")
print(f"Mean density: {np.mean(rho):.1f} kg/m³")
print("=" * 40)

# --------------------------------------------------
# Density cross-section plots - WITH FIXED LAYOUT
# --------------------------------------------------
mean_xz = np.mean(rho, axis=1)
mean_yz = np.mean(rho, axis=0)
mean_xy = np.mean(rho, axis=2)

# Get coordinates for plotting
x_vals_xz = np.mean(x, axis=1)[:, 0]
z_vals_xz = np.mean(z, axis=1)[0, :]
y_vals_yz = np.mean(y, axis=0)[:, 0]
z_vals_yz = np.mean(z, axis=0)[0, :]
x_vals_xy = np.mean(x, axis=2)[:, 0]
y_vals_xy = np.mean(y, axis=2)[0, :]

# Create meshgrids
X_xz, Z_xz = np.meshgrid(x_vals_xz, z_vals_xz, indexing='ij')
Y_yz, Z_yz = np.meshgrid(y_vals_yz, z_vals_yz, indexing='ij')
X_xy, Y_xy = np.meshgrid(x_vals_xy, y_vals_xy, indexing='ij')

# Set colorbar limits
vmin = min(mean_xz.min(), mean_yz.min(), mean_xy.min())
vmax = max(mean_xz.max(), mean_yz.max(), mean_xy.max())

# **FIX 1: Use constrained_layout instead of tight_layout**
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Plot XZ plane
cs1 = axes[0].contourf(X_xz, Z_xz, mean_xz, levels=50, vmin=vmin, vmax=vmax, cmap='viridis')
axes[0].plot(x_bar, z_bar, "xk", markersize=5, markeredgewidth=2)
axes[0].set_title("Mean Density - XZ Plane")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("z [m]")
axes[0].set_aspect('equal')

# Plot YZ plane
cs2 = axes[1].contourf(Y_yz, Z_yz, mean_yz, levels=50, vmin=vmin, vmax=vmax, cmap='viridis')
axes[1].plot(y_bar, z_bar, "xk", markersize=5, markeredgewidth=2)
axes[1].set_title("Mean Density - YZ Plane")
axes[1].set_xlabel("y [m]")
axes[1].set_ylabel("z [m]")
axes[1].set_aspect('equal')

# Plot XY plane
cs3 = axes[2].contourf(X_xy, Y_xy, mean_xy, levels=50, vmin=vmin, vmax=vmax, cmap='viridis')
axes[2].plot(x_bar, y_bar, "xk", markersize=5, markeredgewidth=2)
axes[2].set_title("Mean Density - XY Plane")
axes[2].set_xlabel("x [m]")
axes[2].set_ylabel("y [m]")
axes[2].set_aspect('equal')

# Add single colorbar for all subplots
cbar = fig.colorbar(cs1, ax=axes, shrink=0.8, location='bottom', pad=0.1)
cbar.set_label('Density [kg/m³]')

plt.savefig("density_cross_sections.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# Forward modelling gravity surveys (simplified for testing)
# --------------------------------------------------
print("\n=== FORWARD MODELING (Testing Mode) ===")
print("Note: Using simplified grid for faster testing")

# Use smaller grid for testing
spacing = 25.0  # Larger spacing for testing
xs = np.arange(np.min(x), np.max(x) + spacing, spacing)
ys = np.arange(np.min(y), np.max(y) + spacing, spacing)
X_survey, Y_survey = np.meshgrid(xs, ys)

survey_heights = [0.0, 100.0]  # Just two heights for testing
gz_results = {}

print(f"Survey grid: {X_survey.shape[0]} x {X_survey.shape[1]} points")

# Simplify: Use only significant cells
significant = rho > 0.5 * np.max(rho)
indices = np.where(significant)
print(f"Processing {len(indices[0])} significant cells")

for z_survey in survey_heights:
    gz_grid = np.zeros_like(X_survey)

    # Process cells
    for idx in range(min(100, len(indices[0]))):  # Limit to 100 cells for testing
        i, j, k = indices[0][idx], indices[1][idx], indices[2][idx]
        m_cell = masses[i, j, k]
        xm_cell = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])

        # Vectorized calculation
        for ii in range(X_survey.shape[0]):
            for jj in range(X_survey.shape[1]):
                obs = np.array([X_survey[ii, jj], Y_survey[ii, jj], z_survey])
                gz_grid[ii, jj] += gravity_effect_point(obs, xm_cell, m_cell)

    gz_results[z_survey] = gz_grid
    print(f"✓ z = {z_survey} m: min={gz_grid.min():.2e}, max={gz_grid.max():.2e}")

# --------------------------------------------------
# Plot gz at different elevations - WITH FIXED LAYOUT
# --------------------------------------------------
# **FIX 2: Use GridSpec for better control**
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05])  # Room for colorbar

# Consistent color scale
all_gz = np.concatenate([gz.ravel() for gz in gz_results.values()])
vmin, vmax = all_gz.min(), all_gz.max()

# Plot each survey height
for idx, (z_survey, gz_data) in enumerate(gz_results.items()):
    ax = fig.add_subplot(gs[idx, 0])
    gz_plot = np.where(np.abs(gz_data) > 1e-10, gz_data, np.nan)

    cs = ax.contourf(X_survey, Y_survey, gz_plot, levels=50,
                     vmin=vmin, vmax=vmax, cmap='plasma')
    ax.set_title(f"$g_z$ at z = {z_survey} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Add single colorbar
cbar_ax = fig.add_subplot(gs[:, 2])
fig.colorbar(cs, cax=cbar_ax, label='$g_z$ [m/s²]')

plt.savefig("gz_elevations.png", dpi=300, bbox_inches='tight')
plt.show()


# --------------------------------------------------
# Second vertical derivative - WITH FIXED LAYOUT
# --------------------------------------------------
def second_derivative(arr, spacing):
    d2 = np.zeros_like(arr)
    for i in range(1, arr.shape[0] - 1):
        for j in range(1, arr.shape[1] - 1):
            d2x = (arr[i + 1, j] - 2 * arr[i, j] + arr[i - 1, j]) / (spacing ** 2)
            d2y = (arr[i, j + 1] - 2 * arr[i, j] + arr[i, j - 1]) / (spacing ** 2)
            d2[i, j] = -(d2x + d2y)
    d2[0, :] = d2[-1, :] = d2[:, 0] = d2[:, -1] = np.nan
    return d2


# Calculate derivatives
d2gz_dz2_0 = second_derivative(gz_results[0.0], spacing)

# **FIX 3: Use subplots_adjust instead of tight_layout**
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4)

masked = np.where(~np.isnan(d2gz_dz2_0), d2gz_dz2_0, 0)
cs = axes.contourf(X_survey, Y_survey, masked, levels=50, cmap='RdBu_r')
axes.set_title("$∂²g_z/∂z²$ at z = 0 m")
axes.set_xlabel("x [m]")
axes.set_ylabel("y [m]")
axes.set_aspect('equal')

# Add colorbar with proper spacing
cbar = plt.colorbar(cs, ax=axes, shrink=0.8, pad=0.05)
cbar.set_label('$∂²g_z/∂z²$ [m⁻¹s⁻²]')

plt.savefig("second_vertical_derivative.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("=" * 50)