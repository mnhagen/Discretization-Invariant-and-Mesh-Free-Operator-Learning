import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from FNO2D_def import FNO2D  # your model definition

# =====================
# Configuration
# =====================
h5_path = "/scratch/mnhagen/datasets/the_well/active_matter/data/train/active_matter_L_10.0_zeta_1.0_alpha_-1.0.hdf5"
model_path = "/scratch/mnhagen/models/active_matter/FNO2D_active_matter_v0.pt"

sample_idx = 1    # which trajectory to visualize
device = "cuda:0" if torch.cuda.is_available() else "cpu"

modes1, modes2, width = 16, 16, 64

# =====================
# Load model
# =====================
model = FNO2D(modes1, modes2, width)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# =====================
# Load dataset
# =====================
with h5py.File(h5_path, "r") as f:
    U = f["t1_fields"]["velocity"][:]   # shape (n_traj, n_steps, Nx, Ny, 2)
    X = f["dimensions"]["x"][:]
    Y = f["dimensions"]["y"][:]

U_traj = U[sample_idx]
u0 = U_traj[0]      # initial condition
uT = U_traj[-1]     # final state (ground truth)
Nx, Ny = u0.shape[0], u0.shape[1]

# =====================
# Prepare tensors
# =====================
u_in = torch.tensor(u0[None, :, :, :], dtype=torch.float32).to(device)   # (1, Nx, Ny, 2)
with torch.no_grad():
    u_pred = model(u_in).cpu().numpy().squeeze()  # (Nx, Ny, 2)

u_pred = np.stack(u_pred, axis=-1) if u_pred.ndim == 2 else u_pred

# =====================
# Plot results
# =====================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(f"Active Matter | Sample {sample_idx} | Mapping $u_0 \\to u_T$")

# Plot Vx
axes[0,0].imshow(u0[...,0], cmap='coolwarm', origin='lower')
axes[0,0].set_title("Initial Vx")
axes[0,1].imshow(uT[...,0], cmap='coolwarm', origin='lower')
axes[0,1].set_title("Ground Truth Vx (T)")
axes[0,2].imshow(u_pred[...,0], cmap='coolwarm', origin='lower')
axes[0,2].set_title("Predicted Vx (T)")

# Plot Vy
axes[1,0].imshow(u0[...,1], cmap='coolwarm', origin='lower')
axes[1,0].set_title("Initial Vy")
axes[1,1].imshow(uT[...,1], cmap='coolwarm', origin='lower')
axes[1,1].set_title("Ground Truth Vy (T)")
axes[1,2].imshow(u_pred[...,1], cmap='coolwarm', origin='lower')
axes[1,2].set_title("Predicted Vy (T)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
