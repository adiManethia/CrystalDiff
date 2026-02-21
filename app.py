import streamlit as st
import torch
import numpy as np
import py3Dmol
from stmol import showmol
from src.model import CrystalDiffusionModel

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CrystalDiff: AI Material Designer",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: CONTROLS & INFO ---
with st.sidebar:
    st.title("💎 CrystalDiff Controls")
    
    st.markdown("### 1. Select Chemistry")
    target_atom = st.selectbox(
        "Choose A-Site Cation",
        ["Ca (Calcium)", "Sr (Strontium)", "Ba (Barium)", "Pb (Lead)"],
        index=1,
        help="The large atom in the center of the cage."
    )
    
    st.markdown("### 2. Diffusion Settings")
    steps = st.slider("Denoising Steps", 10, 100, 50, help="More steps = higher quality, but slower.")
    noise_scale = st.slider("Initial Chaos (Noise)", 0.5, 2.0, 1.0, help="Higher noise means the AI has to be more creative.")

    st.divider()
    
    st.markdown("### 🧠 How it Works")
    st.info("""
    **Generative Diffusion:**
    The model starts with random noise (chaos) and iteratively subtracts noise to find a stable crystal structure.
    
    **E(n)-Equivariance:**
    The AI uses a custom Graph Neural Network that respects the laws of physics (rotational symmetry).
    """)
    
    st.markdown("---")
    st.caption("Built with PyTorch & Streamlit by Aditya Mangal. Inspired by DeepMind's work on generative models for materials science.")

# --- MAIN PAGE ---
st.title("💎 CrystalDiff: Generative Material Design")
st.markdown("""
This application uses **Geometric Deep Learning** to hallucinate new stable crystals.
It was trained on the **Materials Project** database to understand the chemical rules of **Perovskite Oxides ($ABO_3$)**.
""")

# Map selection to Atomic Number
atom_map = {
    "Ca (Calcium)": 20, "Sr (Strontium)": 38,
    "Ba (Barium)": 56, "Pb (Lead)": 82
}
selected_z = atom_map[target_atom]
formula_display = f"{target_atom.split()[0]}TiO₃"

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CrystalDiffusionModel()
    try:
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, None

def calculate_metrics(pos, z):
    """Calculates bond lengths to validate physics."""
    # Find Ti (22) and O (8)
    ti_idx = [i for i, atom in enumerate(z) if atom == 22]
    o_idx = [i for i, atom in enumerate(z) if atom == 8]
    
    if not ti_idx or not o_idx: return 0.0
    
    ti_pos = pos[ti_idx[0]]
    dists = []
    for o in o_idx:
        d = np.linalg.norm(ti_pos - pos[o])
        dists.append(d)
    
    return np.mean(dists)

def make_view(pos, z):
    """Creates a 3D molecule view"""
    view = py3Dmol.view(width=800, height=500)
    xyz_str = f"{len(pos)}\nGenerated\n"
    for i in range(len(pos)):
        elem = "O" if z[i] == 8 else "Ti" if z[i] == 22 else target_atom.split()[0]
        xyz_str += f"{elem} {pos[i,0]:.4f} {pos[i,1]:.4f} {pos[i,2]:.4f}\n"
    view.addModel(xyz_str, "xyz")
    # Style: spheres for atoms, sticks for bonds
    view.setStyle({'sphere': {'scale': 0.25}, 'stick': {'radius': 0.1}})
    view.zoomTo()
    return view

# --- APP LOGIC ---
model, device = load_model()

if model is None:
    st.error(" Model weights not found! Please run 'train.py' first.")
    st.stop()

# Layout: Two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🧪 Experiment Setup")
    st.write(f"**Target Material:** {formula_display}")
    st.write(f"**Structure Family:** Cubic Perovskite")
    
    if st.button("✨ Generate Crystal", type="primary", use_container_width=True):
        
        # 1. Setup Data
        z = torch.tensor([selected_z, 22, 8, 8, 8], device=device) # A-Site, Ti, O, O, O
        num_atoms = 5
        
        # Graph connections
        row = torch.repeat_interleave(torch.arange(num_atoms), num_atoms)
        col = torch.arange(num_atoms).repeat(num_atoms)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0).to(device)

        # 2. Diffusion Loop
        x = torch.randn(num_atoms, 3, device=device) * noise_scale
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        dt = 1.0 / steps
        for i in range(steps):
            t_val = 1.0 - (i * dt)
            t_tensor = torch.tensor([[t_val]], device=device)
            
            with torch.no_grad():
                x_pred = model(x, z, t_tensor, edge_index)
            
            # Euler update
            x = x + (x_pred - x) * 0.1
            
            if i % 5 == 0:
                progress_bar.progress(i / steps)
                status.text(f"Denoising... Step {i}/{steps}")
        
        progress_bar.progress(1.0)
        status.success("Done!")
        
        # 3. Store result in session state to keep it on screen
        st.session_state['generated_pos'] = x.numpy()
        st.session_state['generated_z'] = z.numpy()

with col2:
    st.subheader("⚛️ 3D Visualization")
    
    if 'generated_pos' in st.session_state:
        pos = st.session_state['generated_pos']
        z = st.session_state['generated_z']
        
        # Calculate Physics
        avg_bond = calculate_metrics(pos, z)
        
        # Display Metrics
        m1, m2 = st.columns(2)
        m1.metric("Avg Ti-O Bond Length", f"{avg_bond:.3f} Å")
        
        # Validation Logic
        if 1.8 < avg_bond < 2.2:
            m2.success("✅ Physically Valid")
        else:
            m2.warning("⚠️ Unstable Structure")
            
        # Render 3D
        view = make_view(pos, z)
        showmol(view, height=500, width=800)
        
    else:
        st.info("👈 Select your chemistry on the left and click 'Generate Crystal' to start the AI.")
        st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #444; border-radius: 10px; margin-top: 20px;">
            <h1 style="color: #666;">🧊</h1>
            <p style="color: #888;">Waiting for generation...</p>
        </div>
        """, unsafe_allow_html=True)