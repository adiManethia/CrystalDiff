import torch
import torch.nn as nn
from src.layers import EGNNLayer

class TimeEmbedding(nn.Module):
    """
    Converts a time scalar 't' into a vector embedding.
    This allows the neural network to understand the noise level (time step).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim 
        self.linear_1 = nn.Linear(1, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.act = nn.SiLU() # SiLU is standard for diffusion models 

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (Tensor): Time scalars of shape (Batch_Size, 1).
        
        Returns:
            Tensor: Time embeddings of shape (Batch_Size, dim).
        """
        # t is shape (Batch_Size, 1) -> we want to output (Batch_Size, dim)
        x = self.act(self.linear_1(t))
        x = self.linear_2(x)
        return x 

class CrystalDiffusionModel(nn.Module):
    """
    E(n)-Equivariant Diffusion Model for Crystal Generation.
    Predicts the denoised coordinates given a noisy input.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, max_atom_type: int = 100):
        super().__init__()

        # 1. Atom Embedding: Integer -> Vector
        # Maps atomic numbers (e.g., 8 for Oxygen) to a dense vector
        self.atom_embed = nn.Embedding(max_atom_type, hidden_dim) 

        # 2. Time Embedding: Scalar -> Vector
        # Helps the model know if it's looking at pure noise (t=1) or a crystal (t=0)
        self.time_embed = TimeEmbedding(hidden_dim) 

        # 3. Backbone: Stack of Equivariant GNN layers
        # These update both features (h) and positions (x)
        self.layers = nn.ModuleList([
            EGNNLayer(c_in=hidden_dim, c_out=hidden_dim) 
            for _ in range(num_layers)
        ])

        # Note: We don't need a final linear layer for positions because 
        # the EGNN layers update the coordinates 'x' directly at every step.

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x (Tensor): Noisy atom positions. Shape (N, 3).
            z (Tensor): Atomic numbers. Shape (N,).
            t (Tensor): Time step/Noise level. Shape (Batch_Size, 1).
            edge_index (Tensor): Graph connectivity (Adjacency list). Shape (2, E).

        Returns:
            Tensor: Denoised atom positions. Shape (N, 3).
        """

        # 1. Embed Inputs 
        h = self.atom_embed(z)      # (N, hidden_dim)
        t_emb = self.time_embed(t)  # (Batch, hidden_dim)

        # 2. Condition on Time
        # Broadcast time embedding to all atoms in the batch
        # (Assuming single batch or handled externally for simplicity)
        h = h + t_emb.mean(dim=0, keepdim=True) 

        # 3. Message Passing (The "Brain")
        for layer in self.layers:
            # Update features (h) and positions (x) respecting symmetry
            h, x = layer(h, x, edge_index) 

        # Return the updated (denoised) positions
        return x

        