import torch
from mp_api.client import MPRester
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- CONFIGURATION ---
load_dotenv()  # Load .env file for API keys
API_KEY = os.getenv("MPI_API_KEY")

# Save path: project root `data/` folder
repo_root = Path(__file__).resolve().parents[1]
SAVE_PATH = repo_root / "data" / "perovskite_dataset.pt"

def fetch_data(limit=2000):
    """
    Fetches a large dataset of ABO3 Perovskites (5 atoms) for the Foundation Model.
    """
    print(f"Connecting to Materials Project...")

    with MPRester(API_KEY) as mpr:
        # 1. Broad Search: Get all stable materials with 5 atoms
        # We search for materials with exactly 5 sites (atoms) in the unit cell.
        # This implicitly targets ABO3 structures (1+1+3 = 5).
        docs = mpr.materials.summary.search(
            is_stable=True,
            nsites=5, 
            fields=["structure", "material_id", "formula_pretty"]
        )

    print(f"Found {len(docs)} stable 5-atom crystals. Processing...")

    dataset = []
    
    # 2. Filter and Process
    # We want oxygen-containing perovskites generally, but let's keep it broad for now.
    # The 'nsites=5' filter does most of the heavy lifting.
    
    count = 0
    for doc in tqdm(docs):
        if count >= limit:
            break
            
        structure = doc.structure
        formula = doc.formula_pretty
        
        # Heuristic check: Perovskites usually have 3 Oxygens. 
        # This filters out random 5-atom things that aren't Perovskites.
        # (Optional but recommended for cleaner data)
        if "O3" not in formula:
            continue

        # --- TENSOR CREATION ---
        
        # A. Atomic Numbers (Integers) -> The "Identity"
        atomic_numbers = [site.specie.number for site in structure]
        z_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # B. Coordinates (Floats) -> The "Geometry"
        coords = [site.coords for site in structure]
        r_tensor = torch.tensor(coords, dtype=torch.float32)
        
        # C. Center of Mass Correction (CRITICAL for Diffusion)
        # We shift the crystal so its center is at (0,0,0).
        # If we don't do this, the model wastes time learning absolute positions.
        r_tensor = r_tensor - torch.mean(r_tensor, dim=0, keepdim=True)

        # Create Data Object
        data_point = {
            "id": str(doc.material_id),
            "formula": formula,
            "z": z_tensor,   # Features
            "pos": r_tensor  # Positions (Centered)
        }
        
        dataset.append(data_point)
        count += 1

    # 3. Save to Disk
    # Ensure directory exists
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(dataset, SAVE_PATH)
    print(f"✅ Successfully saved {len(dataset)} crystals to {SAVE_PATH}")
    print(f"   (Filtered for 5-atom unit cells containing 'O3')")

if __name__ == "__main__":
    fetch_data(limit=2000)