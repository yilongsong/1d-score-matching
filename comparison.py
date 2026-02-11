import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
N_C_POINTS = 100        # Sampling density along C
EPOCHS = 1000           # Training length
HIDDEN_DIM = 64         # Model size
BATCH_SIZE = 64

# --- 2. DATA GENERATION ---
def get_grid_data(n_c_points, samples_per_point):
    """Generates training data on a fixed grid."""
    # c_grid = np.linspace(0, np.pi, n_c_points) # Fixed distance grid 
    c_grid = np.random.uniform(0, np.pi, n_c_points) # Random sampling from condition space
    c_repeated = np.repeat(c_grid, samples_per_point)
    
    mode_choice = np.random.randint(0, 2, size=len(c_repeated))
    clean_signal = np.sin(c_repeated)
    clean_signal[mode_choice == 1] *= -1
    
    noise = np.random.normal(0, 0.1, size=len(c_repeated))
    x = clean_signal + noise
    c_norm = c_repeated / np.pi # Normalize to [0, 1]
    
    data = np.stack([c_norm, x], axis=1)
    return torch.tensor(data, dtype=torch.float32)

# --- 3. MODEL ARCHITECTURE (Shared) ---
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
        # Handle both integer steps (Diffusion) and float time (Flow)
        if t.dtype == torch.float32:
            # Scale 0-1 float to 0-100 to match rough frequency of steps
            t = t * 100.0
            
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

# --- 4. TRAINING LOOPS ---

def train_diffusion(dataset, name):
    print(f"\n--- Training Diffusion: {name} ---")
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
        
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]
            if len(batch) < 4: continue 
            
            c = batch[:, 0:1]
            x0 = batch[:, 1:2]
            
            t = torch.randint(0, n_steps, (len(batch),), device=device)
            epsilon = torch.randn_like(x0)
            
            a_bar = alpha_bar[t].view(-1, 1)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * epsilon
            
            # Pass t as standard int tensor
            pred = model(x_t, t, c)
            loss = nn.MSELoss()(pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

def train_flow_matching(dataset, name):
    print(f"\n--- Training Flow Matching: {name} ---")
    device = torch.device("cpu")
    model = ConditionalDiffusionMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        indices = torch.randperm(len(dataset))
        dataset = dataset[indices]
        
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]
            if len(batch) < 4: continue 
            
            c = batch[:, 0:1]
            x1 = batch[:, 1:2]          # Target (Data)
            x0 = torch.randn_like(x1)   # Source (Noise)
            
            # 1. Sample t uniform [0, 1]
            t = torch.rand(len(batch), 1, device=device)
            
            # 2. Linear Interpolation (Rectified Flow / Conditional FM)
            # x_t = t * x1 + (1 - t) * x0
            x_t = t * x1 + (1 - t) * x0
            
            # 3. Vector Field Target: v = x1 - x0
            target_v = x1 - x0
            
            # 4. Predict Velocity
            # We flatten t to shape (batch,) for the embedding layer
            pred_v = model(x_t, t.view(-1), c)
            
            loss = nn.MSELoss()(pred_v, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

# --- 5. VISUALIZATION HELPERS ---

def plot_hist_on_ax(ax, c_data, x_data, title):
    h = ax.hist2d(c_data, x_data, bins=[50, 50], range=[[0, 1], [-2, 2]], density=True, cmap='viridis')
    ax.set_title(title, fontsize=10)
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Conditioning')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Prediction')
    else:
        ax.set_yticks([])

def generate_ground_truth(ax):
    n = 100000
    c = np.random.uniform(0, np.pi, n)
    mode_choice = np.random.randint(0, 2, size=n)
    clean = np.sin(c)
    clean[mode_choice == 1] *= -1
    noise = np.random.normal(0, 0.1, size=n)
    x = clean + noise
    c_norm = c / np.pi
    plot_hist_on_ax(ax, c_norm, x, "Ground Truth")

def sample_diffusion(model, ax, title):
    model.eval()
    n_steps = 100
    beta = torch.linspace(1e-4, 0.2, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    n_per_c = 100
    c_grid = torch.linspace(0, 1, 100)
    c_query = c_grid.repeat_interleave(n_per_c).view(-1, 1)
    
    x = torch.randn_like(c_query)
    
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            t = torch.full((len(x),), i, dtype=torch.long)
            pred_noise = model(x, t, c_query)
            
            noise = torch.randn_like(x) if i > 0 else 0
            alpha_t = alpha[i]
            ab_t = alpha_bar[i]
            
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - ab_t)) * pred_noise)
            x = x + torch.sqrt(beta[i]) * noise
            
    plot_hist_on_ax(ax, c_query.flatten().numpy(), x.flatten().numpy(), title)

def sample_flow_matching(model, ax, title):
    model.eval()
    steps = 100
    dt = 1.0 / steps
    
    n_per_c = 100
    c_grid = torch.linspace(0, 1, 100)
    c_query = c_grid.repeat_interleave(n_per_c).view(-1, 1)
    
    # Start from pure noise (x0)
    x = torch.randn_like(c_query)
    
    with torch.no_grad():
        # Euler Integration from t=0 to t=1
        for i in range(steps):
            t_val = i / steps
            t = torch.full((len(x),), t_val, dtype=torch.float32)
            
            # Predict velocity
            v = model(x, t, c_query)
            
            # Step
            x = x + v * dt
            
    plot_hist_on_ax(ax, c_query.flatten().numpy(), x.flatten().numpy(), title)

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Setup Figure: 2 Rows (Diff, Flow), 4 Columns (GT + 3 datasets)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    
    # 1. Plot Ground Truth in first column
    generate_ground_truth(axes[0, 0])
    # axes[0, 0].text(-0.35, 0.5, "Diffusion", transform=axes[0,0].transAxes, 
    #                 fontsize=14, rotation=90, va='center', fontweight='bold')
    
    generate_ground_truth(axes[1, 0])
    # axes[1, 0].text(-0.35, 0.5, "Flow Matching", transform=axes[1,0].transAxes, 
    #                 fontsize=14, rotation=90, va='center', fontweight='bold')
    
    # 2. Experiment Configs
    configs = [
        {"name": "Single-Sample Dataset\nk=1", "k": 1},
        {"name": "Multiple-Sample-Per-Condition Dataset\nk=5", "k": 5},
        {"name": "Multiple-Sample-Per-Condition Dataset\nk=20", "k": 20}
    ]
    
    # 3. Train and Plot
    for i, cfg in enumerate(configs):
        col_idx = i + 1
        dataset = get_grid_data(N_C_POINTS, cfg['k'])
        
        # --- Row 1: Diffusion ---
        diff_model = train_diffusion(dataset, f"Diff {cfg['name']}")
        sample_diffusion(diff_model, axes[0, col_idx], f"Diffusion\n{cfg['name']}")
        
        # --- Row 2: Flow Matching ---
        flow_model = train_flow_matching(dataset, f"Flow {cfg['name']}")
        sample_flow_matching(flow_model, axes[1, col_idx], f"Flow Matching\n{cfg['name']}")

    plt.suptitle(f"Single- v. Multiple-Sample-Per-Condition Dataset Performance", fontsize=14)
    plt.savefig("diffusion_comparison.png")
    plt.show()