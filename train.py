import torch
import torch.optim as optim
import random
import os
from src.model import CrystalDiffusionModel

# --- CONFIGURATION ---
# Check your data folder to ensure filename matches exactly
DATA_PATH = "data/perovskite_dataset.pt" 
EPOCHS = 3000
LEARNING_RATE = 1e-3

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Could not find dataset at {DATA_PATH}. Check spelling!")
    
    data = torch.load(DATA_PATH)
    print(f"✅ Loaded {len(data)} crystals for training.")
    return data

def get_random_batch(dataset, device):
    """
    Picks a RANDOM crystal from the dataset.
    This is crucial for generalization (learning rules vs memorizing one shape).
    """
    # 1. Pick random sample
    sample = random.choice(dataset)
    
    # 2. Extract Data
    z = sample["z"].to(device).long()
    x_real = sample["pos"].to(device).float()
    
    # 3. Build Graph (Fully Connected)
    # We build this dynamically in case crystals have different sizes
    num_atoms = z.shape[0]
    
    # Create all pairs (0,0), (0,1)... (N,N)
    row = torch.repeat_interleave(torch.arange(num_atoms), num_atoms)
    col = torch.arange(num_atoms).repeat(num_atoms)
    
    # Remove self-loops (atoms don't connect to themselves)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0).to(device)
    
    return x_real, z, edge_index

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Training on {device} ---")
    
    # Load Data
    dataset = load_dataset()
    
    # Initialize Model
    model = CrystalDiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print(f"--- Starting Training Loop ({EPOCHS} Epochs) ---")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad() 

        # 2. Get Random Batch
        x_real, z, edge_index = get_random_batch(dataset, device)

        # 3. Diffusion Step (Forward)
        # Sample random time 't' (how much noise to add)
        t = torch.rand(1, 1, device=device) 

        # Create Noise
        noise = torch.randn_like(x_real)
        
        # Add noise: x_noisy = Real + (Noise * t)
        x_noisy = x_real + (noise * t)

        # 4. Model Prediction (Reverse)
        # Predict the denoised structure
        x_pred = model(x_noisy, z, t, edge_index)

        # 5. Calculate Loss
        # We want the predicted position to match the real position
        loss = torch.mean((x_pred - x_real)**2)

        loss.backward()
        optimizer.step()

        # Log progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Save the smarter model
    torch.save(model.state_dict(), "model_weights.pth")
    print("✅ Training Complete. Model saved to model_weights.pth!")

if __name__ == "__main__":
    train()