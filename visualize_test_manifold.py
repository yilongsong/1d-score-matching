import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gaussian_pdf(x, mean, sigma):
    """Computes the value of the Gaussian PDF at x."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def plot_exact_distribution():
    # 1. Create a coordinate grid (No random sampling!)
    # C ranges from 0 to 1 (Normalized), which corresponds to 0 to pi in sine space
    c_norm_vals = np.linspace(0, 1, 100) 
    x_vals = np.linspace(-2.5, 2.5, 100)
    
    # Create the meshgrid (2D arrays for C and X)
    C_GRID, X_GRID = np.meshgrid(c_norm_vals, x_vals)
    
    # 2. Define the physics of the distribution
    sigma = 0.2  # Match the noise level in your generator
    
    # Calculate the two means for every point in the grid
    # Remember: c_norm * pi = original c
    mean_1 = np.sin(C_GRID * np.pi)
    mean_2 = -np.sin(C_GRID * np.pi) # or clean_signal * -1
    
    # 3. Calculate Exact PDF (Mixture of two Gaussians)
    # Z = 0.5 * PDF(Mode1) + 0.5 * PDF(Mode2)
    Z = 0.5 * gaussian_pdf(X_GRID, mean_1, sigma) + \
        0.5 * gaussian_pdf(X_GRID, mean_2, sigma)

    # 4. Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(C_GRID, X_GRID, Z, 
                           cmap=cm.viridis, 
                           edgecolor='none', 
                           alpha=0.9,
                           antialiased=True)

    ax.set_title('Exact Analytical Density (Ground Truth)', fontsize=14)
    ax.set_xlabel('Condition (Normalized)', fontsize=12, labelpad=10)
    ax.set_ylabel('Target (x)', fontsize=12, labelpad=10)
    ax.set_zlabel('Probability Density', fontsize=12, labelpad=10)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

    # Adjust view
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_exact_distribution()