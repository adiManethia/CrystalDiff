import torch
from src.layers import EGNNLayer

def get_dummy_graph():
    # 3 Atoms
    h = torch.randn(3, 16) # Random features
    x = torch.randn(3, 3)  # Random positions
    
    # Fully connected graph (0-1, 1-2, 2-0, etc.)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], dtype=torch.long)
    
    return h, x, edge_index

def test_equivariance():
    print("--- Testing Equivariance ---")
    
    # 1. Initialize Layer
    layer = EGNNLayer(c_in=16, c_out=16)
    
    # 2. Get Data
    h, x, edge_index = get_dummy_graph()
    
    # 3. Create a Rotation Matrix (90 degrees around Z axis)
    # This simulates rotating the crystal
    rot = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    
    # 4. Scenario A: Pass original data
    h_out_1, x_out_1 = layer(h, x, edge_index)
    
    # Apply rotation to the OUTPUT of Scenario A
    x_out_1_rotated = x_out_1 @ rot
    
    # 5. Scenario B: Rotate INPUT, then pass through layer
    x_rotated = x @ rot
    h_out_2, x_out_2 = layer(h, x_rotated, edge_index)
    
    # 6. Compare!
    # Does Rotate(Model(x)) == Model(Rotate(x)) ?
    diff = torch.abs(x_out_1_rotated - x_out_2).max()
    
    print(f"Difference between methods: {diff.item():.6f}")
    
    if diff < 1e-5:
        print("✅ SUCCESS: The layer is Equivariant!")
    else:
        print("❌ FAIL: The layer is NOT Equivariant.")

if __name__ == "__main__":
    test_equivariance()