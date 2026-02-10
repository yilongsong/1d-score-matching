
import torch
import torch.nn as nn
import math

class ConditionalDiffusionMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.time_dim = 32 # sinusoidal

        input_dim = 1 + 1 + self.time_dim # 1 (x_t) + 1 (cond) + 32 (time)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def get_sinusoidal_embeddings(self, t): ### !!!
        # Standard transformer-style embeddings
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, c):
        # x: (batch, 1), t: (batch), c: (batch, 1)
        
        # Embed time
        t_emb = self.get_sinusoidal_embeddings(t)
        
        # Concatenate x, condition, and time embedding
        # Shape becomes (batch, 1 + 1 + 32)
        inp = torch.cat([x, c, t_emb], dim=1)
        
        # Return noise prediction
        return self.net(inp)