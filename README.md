# 💎 CrystalDiff: Conditional Generative AI for Material Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**A geometric deep learning framework for the inverse design of 3D crystal structures using E(n)-Equivariant Denoising Diffusion Probabilistic Models (DDPM).**

![Streamlit Web App](cry_test.png)
*(CrystalDiff interactive web application generating novel Perovskite structures in real-time.)*

---

## 🔬 Project Overview

The discovery of novel materials (e.g., for solid-state batteries or photovoltaics) is traditionally bottlenecked by the computational cost of Density Functional Theory (DFT). **CrystalDiff** bypasses this by using Generative AI to "dream" chemically valid, stable structures in milliseconds.

Moving beyond simple property prediction, this project implements a **Conditional Generative Foundation Model** trained on the Materials Project database. It allows users to condition the generation process on desired macroscopic properties (like Band Gap) and target specific atomic compositions.

### Key Features
* **Geometric Deep Learning from Scratch:** Implements custom **E(n)-Equivariant Message Passing** layers in raw PyTorch to guarantee physical rotational invariance without relying on high-level graph wrappers.
* **Property-Conditioned Diffusion:** Injects target physical properties (e.g., Band Gap) into the reverse diffusion process to solve the "Inverse Design" problem.
* **Scientific Validation:** Achieves realistic physical metrics, successfully learning Pauli Exclusion and covalent bond lengths (e.g., Ti-O at ~1.9 Å) purely from coordinate data.
* **Interactive Web Interface:** Includes a fully functional Streamlit application for real-time 3D crystal generation, validation, and visualization.

---

## 🏗 Architecture & Mathematics

The core model relies on a Time-and-Property-Conditioned Equivariant Graph Neural Network (EGNN).

### 1. The Forward Process (Data $\to$ Noise)
We progressively corrupt real crystal coordinate structures $x_0$ by adding Gaussian noise $\epsilon$ over $T$ timesteps:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

### 2. The Conditional Reverse Process (Noise $\to$ Data)
The neural network learns to predict the noise $\epsilon_\theta$. Crucially, it is conditioned on both the time step embedding $t$ and a target physical property embedding $y$ (e.g., Band Gap):

$$x_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t, y) \right) + \sigma_t z$$

### 3. E(n)-Equivariance (The Physics Layer)
To respect physical symmetry (rotating a crystal does not change its internal energy or chemistry), the node positions are updated via covariant vector steps, scaled by invariant edge messages $m_{ij}$:

$$x_i^{l+1} = x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_{pos}(m_{ij})$$

---

## 📊 Scientific Validation

To prove the model learned actual chemistry and not just random point cloud distributions, we analyzed the **Radial Distribution Function (RDF)** of the generated crystals.

![Radial Distribution Function Analysis](rdf_analysis.png)

**Key Findings:**
1. **Pauli Exclusion Principle (0.0 to 1.5 Å):** The probability density is strictly zero, proving the model learned that atoms cannot physically overlap (unlike the random noise baseline).
2. **Covalent Bonding (~1.9 Å):** The first sharp peak aligns perfectly with standard Titanium-Oxygen bond lengths, demonstrating the model learned local chemical environments from scratch.
3. **Lattice Formation:** Secondary peaks (> 2.5 Å) confirm the generation of long-range repeating crystalline order.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* Materials Project API Key (`mp-api`)

### 1. Clone & Install
```bash
git clone [https://github.com/YOUR_USERNAME/CrystalDiff.git](https://github.com/YOUR_USERNAME/CrystalDiff.git)
cd CrystalDiff
pip install -r requirements.txt