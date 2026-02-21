import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import CrystalDiffusionModel

# Load your model and generate a crystal
def generate_crystal():
    model = CrystalDiffusionModel()
    model.load_state_dict(torch.load("model_weights.pth", map_location='cpu'))
    model.eval()
    
    # Generate 5 atoms
    num_atoms = 5
    z = torch.tensor([56, 22, 8, 8, 8]) # BaTiO3
    
    # Graph Setup
    row = torch.repeat_interleave(torch.arange(num_atoms), num_atoms)
    col = torch.arange(num_atoms).repeat(num_atoms)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    
    # Diffusion
    x = torch.randn(num_atoms, 3) # Start with noise
    steps = 50
    dt = 1.0 / steps
    
    for i in range(steps):
        t = torch.tensor([[1.0 - i*dt]])
        with torch.no_grad():
            pred = model(x, z, t, edge_index)
        x = x + (pred - x) * 0.1
        
    return x.numpy()

def compute_rdf(coords, box_size=5.0, bins=50):
    """
    Calculates the Radial Distribution Function (RDF).
    """
    distances = []
    num_atoms = len(coords)
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances.append(dist)
            
    # Histogram
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, box_size))
    r = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize (Volume correction)
    dr = bin_edges[1] - bin_edges[0]
    volume = 4 * np.pi * r**2 * dr
    rdf = hist / (volume * num_atoms) # Density normalization
    
    return r, rdf

def plot_comparison():
    print("Generating Analysis Plot...")
    
    # 1. Get RDF for Random Noise
    noise = np.random.randn(5, 3)
    r_noise, rdf_noise = compute_rdf(noise)
    
    # 2. Get RDF for Generated Crystal
    crystal = generate_crystal()
    r_crys, rdf_crys = compute_rdf(crystal)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_noise, rdf_noise, label='Random Noise', linestyle='--', color='gray')
    plt.plot(r_crys, rdf_crys, label='Generated Crystal (AI)', linewidth=3, color='blue')
    
    plt.title("Radial Distribution Function (RDF) Analysis")
    plt.xlabel("Distance (Angstroms)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("rdf_analysis.png")
    print("✅ Saved 'rdf_analysis.png'. Put this in your README!")

if __name__ == "__main__":
    plot_comparison()