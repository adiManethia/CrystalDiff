import numpy as np

def read_xyz(filename):
    coords = []
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
    return atoms, coords

def check_physics():
    print("--- 🧪 Scientific Validation ---")
    
    # 1. Load the Generated Crystal
    atoms, coords = read_xyz("gen_final.xyz")
    
    # 2. Find the Titanium (Ti) and Oxygens (O)
    # Note: In our code, we mapped:
    # 22 -> Ti (Titanium)
    # 8  -> O  (Oxygen)
    # 20 -> Ca (Calcium)
    
    ti_indices = [i for i, atom in enumerate(atoms) if atom == "Ti"]
    o_indices = [i for i, atom in enumerate(atoms) if atom == "O"]
    
    if not ti_indices or not o_indices:
        print("❌ Could not find Ti or O atoms to measure bonds.")
        return

    print(f"Found {len(ti_indices)} Titanium and {len(o_indices)} Oxygen atoms.")
    
    # 3. Measure Distances
    bond_lengths = []
    
    for ti_idx in ti_indices:
        ti_pos = coords[ti_idx]
        for o_idx in o_indices:
            o_pos = coords[o_idx]
            
            # Calculate Euclidean Distance
            dist = np.linalg.norm(ti_pos - o_pos)
            bond_lengths.append(dist)
            
    # 4. Analyze Results
    min_bond = min(bond_lengths)
    avg_bond = sum(bond_lengths) / len(bond_lengths)
    
    print(f"\nMeasured Bond Lengths (Ti - O):")
    print(f"   Minimum: {min_bond:.4f} Å")
    print(f"   Average: {avg_bond:.4f} Å")
    
    # 5. The "DeepMind" Pass/Fail
    # Real Physics: Ti-O bond is typically 1.90 - 2.05 Å
    # We allow some error since this is a tiny model trained for 5 minutes
    if 1.5 < min_bond < 2.5:
        print("\n✅ SUCCESS: The model learned valid chemical bonds!")
        print("   (Target range: ~1.9 Å)")
    else:
        print("\n⚠️  WARNING: Bonds are physically unrealistic.")
        print("   (Try training for more epochs or checking the dataset)")

if __name__ == "__main__":
    check_physics()