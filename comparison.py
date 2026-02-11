import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. CONFIGURATION ---
N_C_POINTS = 100        # Sampling density along C
EPOCHS = 1000           # Training length
HIDDEN_DIM = 64        # Model size
BATCH_SIZE = 64

# --- 2. DATA GENERATION ---
def get_grid_data(n_c_points, samples_per_point):
    """Generates training data on a fixed grid."""
    c_grid = np.linspace(0, np.pi, n_c_points)
    c_repeated = np.repeat(c_grid, samples_per_point)
    
    mode_choice = np.random.randint(0, 2, size=len(c_repeated))
    clean_signal = np.sin(c_repeated)
    clean_signal[mode_choice == 1] *= -1
    
    noise = np.random.normal(0, 0.1, size=len(c_repeated))
    x = clean_signal + noise
    c_norm = c_repeated / np.pi # Normalize to [0, 1]
    
    data = np.stack([c_norm, x], axis=1)
    return torch.tensor(data, dtype=torch.float32)

# --- 3. MODEL ARCHITECTURE ---
class ConditionalDiffusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_dim = 16
        input_dim = 1 + 1 + self.time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def get_sinusoidal_embeddings(self, t):
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, c):
        t_emb = self.get_sinusoidal_embeddings(t)
        inp = torch.cat([x, c, t_emb], dim=1)
        return self.net(inp)

# --- 4. TRAINING LOOP ---
def train_model(dataset, name):
    print(f"\n--- Training {name} (Size: {len(dataset)}) ---")
    device = torch.device("cpu")
    model = ConditionalDiffusionMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    n_steps = 100
    beta = torch.linspace(1e-4, 0.2, n_steps).to(device)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    for epoch in range(EPOCHS):
        indices = torch.randperm(len(dataset))
        dataset = dataset[indices]
        
        # Simple batch loop
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]
            if len(batch) < 4: continue 
            
            c = batch[:, 0:1]
            x0 = batch[:, 1:2]
            
            t = torch.randint(0, n_steps, (len(batch),), device=device)
            epsilon = torch.randn_like(x0)
            
            a_bar = alpha_bar[t].view(-1, 1)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * epsilon
            
            pred = model(x_t, t, c)
            loss = nn.MSELoss()(pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

# --- 5. VISUALIZATION HELPERS ---
def plot_hist_on_ax(ax, c_data, x_data, title):
    """Standardized plotting function for all panels"""
    h = ax.hist2d(c_data, x_data, bins=[50, 50], range=[[0, 1], [-2, 2]], density=True, cmap='viridis')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Conditioning')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Prediction')
    else:
        ax.set_yticks([])

def generate_ground_truth(ax):
    """Generates a massive amount of data from the true distribution formula."""
    n = 100000
    c = np.random.uniform(0, np.pi, n)
    mode_choice = np.random.randint(0, 2, size=n)
    clean = np.sin(c)
    clean[mode_choice == 1] *= -1
    noise = np.random.normal(0, 0.1, size=n)
    x = clean + noise
    c_norm = c / np.pi
    
    plot_hist_on_ax(ax, c_norm, x, "Ground Truth\n(Target Density)")

def sample_model(model, ax, title):
    """Samples from the trained diffusion model."""
    model.eval()
    n_steps = 100
    beta = torch.linspace(1e-4, 0.2, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Query grid: We want to sample uniformly across C to see the manifold
    n_per_c = 100
    c_grid = torch.linspace(0, 1, 100) # 100 points along C
    c_query = c_grid.repeat_interleave(n_per_c).view(-1, 1) # 10,000 total points
    
    x = torch.randn_like(c_query)
    
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            t = torch.full((len(x),), i, dtype=torch.long)
            pred_noise = model(x, t, c_query)
            
            # DDPM Update
            noise = torch.randn_like(x) if i > 0 else 0
            alpha_t = alpha[i]
            ab_t = alpha_bar[i]
            
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - ab_t)) * pred_noise)
            x = x + torch.sqrt(beta[i]) * noise
            
    plot_hist_on_ax(ax, c_query.flatten().numpy(), x.flatten().numpy(), title)

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Setup Figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    
    # 1. Plot Ground Truth
    generate_ground_truth(axes[0])
    
    # 2. Experiment Configs
    configs = [
        {"name": "Single-Sample Dataset\n(k=1, Sparse)", "k": 1},
        {"name": "Multiple-Sample-Per-Condition Dataset\n(k=5)", "k": 5},
        {"name": "Multiple-Sample-Per-Condition Dataset\n(k=20)", "k": 20}
    ]
    
    # 3. Train and Plot
    for i, cfg in enumerate(configs):
        dataset = get_grid_data(N_C_POINTS, cfg['k'])
        model = train_model(dataset, cfg['name'])
        
        # Plot into axes 1, 2, 3
        sample_model(model, axes[i+1], cfg['name'])

    plt.suptitle(f"Single- v. Multiple-Sample-Per-Condition Dataset Performance", fontsize=14)
    plt.savefig("diffusion_comparison.png")
    plt.show()