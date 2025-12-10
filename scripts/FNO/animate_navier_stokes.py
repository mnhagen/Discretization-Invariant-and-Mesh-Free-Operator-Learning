"""
Animate 2D Navier–Stokes vorticity evolution
--------------------------------------------

Generates a single trajectory ω(x,y,t) using the same pseudo-spectral RK4 solver
from the dataset generator, and creates an animation of vorticity over time.

Author: ChatGPT :)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ===========================================================
# Parameters (can be changed)
# ===========================================================
N = 64           # grid resolution
L = 1.0          # domain size
dt = 1e-3        # time step
nu = 1e-5        # viscosity
t_final = 5.0   # total simulated time (seconds)
store_every = 20 # store every N steps to control frame rate
kmax = 12         # max frequency for random initial condition

# ===========================================================
# Helper functions (same as in generator)
# ===========================================================
def random_vorticity_ic(N, kmax):
    """Smooth random vorticity field via truncated Fourier modes."""
    coeffs = np.zeros((N, N), dtype=np.complex128)
    for i in range(1, kmax):
        for j in range(1, kmax):
            amp = np.exp(-0.5 * ((i**2 + j**2) / (kmax / 2)**2))
            coeffs[i, j] = amp * (np.random.randn() + 1j * np.random.randn())
    coeffs[-kmax + 1 :, -kmax + 1 :] = np.conj(np.flip(np.flip(coeffs[1:kmax, 1:kmax], 0), 1))
    w = np.real(np.fft.ifft2(coeffs))
    w /= np.max(np.abs(w))
    return w

def velocity_from_vorticity(w_hat, kx, ky, ksq):
    """Compute velocity u,v from vorticity via streamfunction."""
    psi_hat = -w_hat / ksq
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = np.real(np.fft.ifft2(u_hat))
    v = np.real(np.fft.ifft2(v_hat))
    return u, v

def nonlinear_term(w_hat, kx, ky, ksq):
    """Compute -u·∇w term."""
    u, v = velocity_from_vorticity(w_hat, kx, ky, ksq)
    w_x = np.real(np.fft.ifft2(1j * kx * w_hat))
    w_y = np.real(np.fft.ifft2(1j * ky * w_hat))
    adv = u * w_x + v * w_y
    return -np.fft.fft2(adv)

def ns_rk4_forced(w0, nu, dt, N_t, store_every, kx, ky, ksq, forcing_amp=0.1):
    w_hat = np.fft.fft2(w0)
    expLdt = np.exp(-nu * ksq * dt)
    expLdt2 = np.exp(-nu * ksq * dt / 2)

    # Precompute forcing in spectral space
    y = np.linspace(0, 1, w0.shape[0], endpoint=False)
    Y = np.tile(y, (w0.shape[1], 1)).T
    f = forcing_amp * np.sin(4 * np.pi * Y)  # f(x,y) = 0.1 sin(4y)
    f_hat = np.fft.fft2(f)

    snapshots = [np.real(np.fft.ifft2(w_hat))]
    for n in range(1, N_t + 1):
        k1 = nonlinear_term(w_hat, kx, ky, ksq) + f_hat
        k2 = nonlinear_term(expLdt2 * (w_hat + 0.5 * dt * k1), kx, ky, ksq) + f_hat
        k3 = nonlinear_term(expLdt2 * (w_hat + 0.5 * dt * k2), kx, ky, ksq) + f_hat
        k4 = nonlinear_term(expLdt * (w_hat + dt * k3), kx, ky, ksq) + f_hat
        w_hat = expLdt * w_hat + (dt / 6.0) * expLdt * (k1 + 2*k2 + 2*k3 + k4)
        if n % store_every == 0:
            snapshots.append(np.real(np.fft.ifft2(w_hat)))
    return np.stack(snapshots, axis=0)

# ===========================================================
# Simulation
# ===========================================================
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
kx, ky = np.meshgrid(kx, ky, indexing="ij")
ksq = kx**2 + ky**2
ksq[0, 0] = 1.0

N_t = int(t_final / dt)

print(f"Simulating Navier–Stokes 2D: N={N}, ν={nu}, t_final={t_final}, steps={N_t}")

w0 = random_vorticity_ic(N, kmax)
w_traj = ns_rk4_forced(w0, nu, dt, N_t, store_every, kx, ky, ksq)

print(f"Generated {w_traj.shape[0]} frames")

# ===========================================================
# Animation
# ===========================================================
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(w_traj[0], cmap="coolwarm", origin="lower", extent=[0, L, 0, L])
ax.set_title("2D Navier–Stokes Vorticity")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    im.set_data(w_traj[frame])
    ax.set_title(f"t = {frame * dt * store_every:.2f} s")
    return [im]

anim = FuncAnimation(fig, update, frames=w_traj.shape[0], interval=50, blit=True)

# Save as GIF in your scratch directory
from matplotlib.animation import FFMpegWriter
import os

save_dir = "/scratch/mnhagen/animations/navier_stokes"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "navier_stokes_vorticity_forced4.mp4")
writer = FFMpegWriter(fps=30, bitrate=1800)
anim.save(save_path, writer=writer)

print(f"Saved animation to: {save_path}")
