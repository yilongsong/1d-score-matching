import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Reuse the generator logic from before
def get_data(n_samples=50000): # Increased samples for smoother 3D density
    c = np.random.uniform(-2 * np.pi, 2 * np.pi, n_samples)
    mode_choice = np.random.randint(0, 2, n_samples)
    clean_signal = np.sin(c)
    clean_signal[mode_choice == 1] *= -1
    noise = np.random.normal(0, 0.1, n_samples)
    x = clean_signal + noise
    c_norm = c / (2 * np.pi)
    return c_norm, x

def plot_3d_distribution():
    # 1. Generate lots of data to get a good density estimate
    c, x = get_data()

    # 2. Create a 2D Histogram (Heatmap logic)
    # This counts how many data points fall into each grid square
    bins = 60
    hist, c_edges, x_edges = np.histogram2d(c, x, bins=bins, density=True)

    # 3. Prepare coordinates for 3D plotting
    # We need a grid of (c, x) coordinates
    c_centers = (c_edges[:-1] + c_edges[1:]) / 2
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    C_GRID, X_GRID = np.meshgrid(c_centers, x_centers, indexing='ij')

    # 4. Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    # 'cmap' colors higher peaks differently
    surf = ax.plot_surface(C_GRID, X_GRID, hist, 
                           cmap=cm.viridis, 
                           edgecolor='none', 
                           alpha=0.9,
                           rstride=1, cstride=1)

    # Add labels
    ax.set_title('The "Mirror Sines" Probability Manifold', fontsize=14)
    ax.set_xlabel('Conditioning Signal (c)', fontsize=12, labelpad=10)
    ax.set_ylabel('Target Value (x)', fontsize=12, labelpad=10)
    ax.set_zlabel('Probability Density', fontsize=12, labelpad=10)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

    # Adjust view angle for best look
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_distribution()