import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_xyz(filename):
    """
    Reads an XYZ file and returns coordinates and atom types.
    """
    coords = []
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip header lines (first 2)
        for line in lines[2:]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords), atoms

def plot_crystal(ax, coords, atoms, title):
    """
    Plots a single crystal in a 3D subplot.
    """
    # Define colors for atoms (Titanium=Silver, Oxygen=Red, Ca=Green)
    colors = {'Ti': 'gray', 'O': 'red', 'Ca': 'green', 'Pb': 'black', 'I': 'purple'}
    
    # Scatter plot
    # s=size of atom, alpha=transparency
    for i, atom in enumerate(atoms):
        color = colors.get(atom, 'blue') # Default to blue if unknown
        ax.scatter(coords[i,0], coords[i,1], coords[i,2], 
                  c=color, s=200, edgecolors='k', alpha=0.8)
    
    # Draw "bonds" (lines between atoms close to each other)
    # This helps visualize the structure
    num_atoms = len(coords)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            # If atoms are closer than 2.8 Angstroms, draw a line
            if dist < 2.8:
                ax.plot([coords[i,0], coords[j,0]], 
                        [coords[i,1], coords[j,1]], 
                        [coords[i,2], coords[j,2]], 
                        c='black', linewidth=1, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set consistent limits so we can compare
    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)
    ax.set_zlim(-2, 5)

def create_comparison_figure():
    # 1. Read Data
    # Make sure you ran generate.py first to get these files!
    noise_pos, atoms = read_xyz("gen_step_00.xyz")
    final_pos, _ = read_xyz("gen_final.xyz")
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 6))
    
    # Plot 1: The Noise
    ax1 = fig.add_subplot(121, projection='3d')
    plot_crystal(ax1, noise_pos, atoms, "Step 0: Random Noise")
    
    # Plot 2: The Generated Crystal
    ax2 = fig.add_subplot(122, projection='3d')
    plot_crystal(ax2, final_pos, atoms, "Step 50: Generated Crystal")
    
    plt.tight_layout()
    plt.savefig("result_plot.png", dpi=300)
    print("Saved comparison figure to 'result_plot.png'")
    plt.show()

if __name__ == "__main__":
    create_comparison_figure()