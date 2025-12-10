"""
1D Burgers dataset generator for autoregressive FNO training
------------------------------------------------------------

Generates time-resolved trajectories u(x,t) using a spectral RK4 solver.
Saves sub-sampled states every `store_every` solver steps so the dataset
remains compact while supporting sequential (autoregressive) training.

Each sample = one trajectory with many (u_t -> u_{t+Δt}) pairs.

Author: <your name>
"""

import numpy as np
import h5py
import os
import timeit


def generate_burgers_autoreg(
    N_x=1024,
    L_x=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=0.01,
    num_samples=100,
    kmax=16,
    mu=1.0,
    store_every=5,
    save_dir="/scratch/mnhagen/datasets/burgers_autoreg/",
    filename=None,
    dtype=np.float32,
):
    """
    Generate 1D Burgers' equation trajectories for autoregressive training.

    Parameters
    ----------
    N_x : int
        Number of spatial grid points.
    L_x : float
        Length of spatial domain [0, L_x).
    L_t : float
        Total simulated time.
    dt : float
        Solver time step.
    nu : float
        Viscosity.
    num_samples : int
        Number of random trajectories to generate.
    kmax : int
        Number of low-frequency Fourier modes for random IC.
    mu : float
        Nonlinear coefficient.
    store_every : int
        Save every `store_every` solver steps (controls Δt_eff = store_every*dt).
    save_dir : str
        Directory where the HDF5 file will be stored.
    filename : str or None
        Optional filename. Auto-generated if None.
    dtype : np.dtype
        Floating point precision to store (np.float32 or np.float16).
    """

    t_start = timeit.default_timer()

    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"burgers1D_autoreg_Nx{N_x}_nu{nu}_samples{num_samples}_dt{dt}_store{store_every}.h5"
    save_path = os.path.join(save_dir, filename)

    dx = L_x / N_x
    X = np.linspace(0, L_x, N_x, endpoint=False)
    N_t = int(L_t / dt)
    dt_eff = store_every * dt

    print(f"\n=== Burgers autoreg dataset ===")
    print(f"Nx={N_x}, Lx={L_x}, L_t={L_t}, dt={dt}, store_every={store_every} -> Δt_eff={dt_eff}")
    print(f"nu={nu}, num_samples={num_samples}")
    print(f"Saving to: {save_path}\n")

    # -------------------------------------------------------------------------
    # Helper: random smooth initial condition
    # -------------------------------------------------------------------------
    def random_initial_condition(N_x, kmax=N_x // 4):
        """Smooth random initial condition via truncated Fourier series."""
        coeffs = np.zeros(N_x, dtype=np.complex128)
        coeffs[1:kmax] = (
            np.random.randn(kmax - 1) + 1j * np.random.randn(kmax - 1)
        ) * np.exp(-0.5 * (np.arange(1, kmax) / (kmax / 4)) ** 2)
        coeffs[-kmax + 1 :] = np.conj(np.flip(coeffs[1:kmax]))
        u = np.real(np.fft.ifft(coeffs))
        u = u / np.max(np.abs(u))  # normalize amplitude
        return u

    # -------------------------------------------------------------------------
    # Solver: pseudo-spectral RK4 with integrating factor
    # -------------------------------------------------------------------------
    def burgers_rk4(u0, mu, nu, k, dt, N_t, store_every):
        u_hat = np.fft.fft(u0)
        N_x = len(u0)
        expLdt = np.exp(-nu * (np.abs(k) ** 2) * dt)
        expLdt2 = np.exp(-nu * (np.abs(k) ** 2) * dt / 2)

        def nonlinear(u_hat_local):
            u = np.real(np.fft.ifft(u_hat_local))
            u_x = np.real(np.fft.ifft(k * u_hat_local))
            return -mu * np.fft.fft(u * u_x)

        snapshots = [np.real(np.fft.ifft(u_hat))]
        for n in range(1, N_t + 1):
            k1 = nonlinear(u_hat)
            k2 = nonlinear(expLdt2 * (u_hat + 0.5 * dt * k1))
            k3 = nonlinear(expLdt2 * (u_hat + 0.5 * dt * k2))
            k4 = nonlinear(expLdt * (u_hat + dt * k3))
            u_hat = expLdt * u_hat + (dt / 6.0) * expLdt * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft(u_hat)))

        return np.stack(snapshots, axis=0)  # shape (T_sub, N_x)

    # -------------------------------------------------------------------------
    # Main generation loop
    # -------------------------------------------------------------------------
    k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx) * 1j
    U_list = []

    for j in range(num_samples):
        u0 = random_initial_condition(N_x, kmax=kmax)
        U_sub = burgers_rk4(u0, mu, nu, k, dt, N_t, store_every)
        U_list.append(U_sub.astype(dtype))
        print(
            f"Sample {j+1}/{num_samples}: stored {U_sub.shape[0]} frames "
            f"(min={U_sub.min():.3f}, max={U_sub.max():.3f})"
        )

    # Determine maximum sequence length
    max_T_sub = max(U.shape[0] for U in U_list)
    T_lengths = np.array([U.shape[0] for U in U_list], dtype=np.int32)

    # Pack into a single array
    U_packed = np.zeros((num_samples, max_T_sub, N_x), dtype=dtype)
    for j, U_sub in enumerate(U_list):
        U_packed[j, : U_sub.shape[0], :] = U_sub

    # -------------------------------------------------------------------------
    # Save to HDF5
    # -------------------------------------------------------------------------
    with h5py.File(save_path, "w") as f:
        f.create_dataset("U", data=U_packed, compression="gzip")
        f.create_dataset("X", data=X.astype(dtype))
        f.create_dataset("T_lengths", data=T_lengths)
        f.attrs["dt"] = dt
        f.attrs["store_every"] = store_every
        f.attrs["dt_eff"] = dt_eff
        f.attrs["nu"] = nu
        f.attrs["L_x"] = L_x
        f.attrs["L_t"] = L_t
        f.attrs["num_samples"] = num_samples
        f.attrs["description"] = (
            "1D Burgers autoregressive dataset: U[sample, time, x]"
        )

    size_mb = os.path.getsize(save_path) / (1024**2)
    print(f"\nSaved dataset: {save_path} ({size_mb:.2f} MB)")
    print(f"U shape: {U_packed.shape}, X shape: {X.shape}")

    print(f"Total generation time: {timeit.default_timer() - t_start:.1f}s")
    print("==========================================\n")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    generate_burgers_autoreg(
        N_x=1024,
        L_x=1.0,
        L_t=1.0,
        dt=1e-3,
        nu=0.01,
        num_samples=100,
        kmax=16,
        store_every=5,  # save every 5 solver steps (Δt_eff = 5e-3)
        dtype=np.float16,  # halve storage size
        save_dir="/scratch/mnhagen/datasets/burgers",
    )
