import numpy as np
import h5py
import os
import timeit


def generate_navier_stokes_autoreg(
    N=64,
    L=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=1e-3,
    num_samples=20,
    kmax=8,
    store_every=10,
    save_dir="/scratch/mnhagen/datasets/navier_stokes_autoreg/",
    filename=None,
    dtype=np.float32,
    forced=True,
    forcing_amp = 0.5,
    spectral_filter_K0 = None
):
    """
    Generate 2D incompressible Navier–Stokes trajectories in vorticity form.

    Parameters
    ----------
    N : int
        Grid resolution (N x N).
    L : float
        Domain length in both x and y (periodic box [0,L)x[0,L)).
    L_t : float
        Total simulated time.
    dt : float
        Time step.
    nu : float
        Viscosity.
    num_samples : int
        Number of random trajectories to generate.
    kmax : int
        Truncation frequency for random initial conditions.
    store_every : int
        Save every `store_every` solver steps (Δt_eff = store_every * dt).
    save_dir : str
        Directory where the HDF5 file will be stored.
    filename : str or None
        Optional filename; auto-generated if None.
    dtype : np.dtype
        Floating point precision for storage.
    forced : bool
        Whether to include sinusoidal forcing in the simulation.
    """

    t_start = timeit.default_timer()

    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = (
            f"navier_stokes2D_autoreg_N{N}_nu{nu}_samples{num_samples}_dt{dt}_store{store_every}"
            + (f"_forced{forcing_amp}" if forced else "") + 
            (f"_filter{spectral_filter_K0}" if spectral_filter_K0 else "")
            + ".h5"
        )
    save_path = os.path.join(save_dir, filename)

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    N_t = int(L_t / dt)
    dt_eff = store_every * dt

    print(f"\n=== Navier–Stokes 2D autoreg dataset ===")
    print(f"N={N}, L={L}, L_t={L_t}, dt={dt}, store_every={store_every} -> Δt_eff={dt_eff}")
    print(f"nu={nu}, num_samples={num_samples}, forced={forced}")
    print(f"Saving to: {save_path}\n")

    if spectral_filter_K0 is not None:
        mask = np.zeros((N, N), dtype=bool)
        K0 = spectral_filter_K0
        # keep only central K0 modes in each direction
        mask[:K0, :K0] = True
        mask[-K0:, :K0] = True
        mask = mask.astype(float)  # 1.0 inside retained region, 0.0 elsewhere
    else:
        mask = None

    # -------------------------------------------------------------------------
    # Helper: random smooth vorticity field for initial condition
    # -------------------------------------------------------------------------
    def random_vorticity_ic(N, kmax):
        """Generate a smooth random vorticity field via truncated Fourier series."""
        coeffs = np.zeros((N, N), dtype=np.complex128)
        for i in range(1, kmax):
            for j in range(1, kmax):
                phase = np.random.rand() * 2 * np.pi
                amp = np.exp(-0.5 * ((i**2 + j**2) / (kmax / 2) ** 2))
                coeffs[i, j] = amp * (np.random.randn() + 1j * np.random.randn())
        coeffs[-kmax + 1 :, -kmax + 1 :] = np.conj(np.flip(np.flip(coeffs[1:kmax, 1:kmax], 0), 1))
        ω = np.real(np.fft.ifft2(coeffs))
        ω /= np.max(np.abs(ω))  # normalize
        return ω

    # -------------------------------------------------------------------------
    # Solver setup
    # -------------------------------------------------------------------------
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    ksq = kx**2 + ky**2
    ksq[0, 0] = 1.0  # avoid division by zero for mean mode

    def velocity_from_vorticity(ω_hat):
        """Compute velocity field u=(u_x,u_y) from vorticity via streamfunction ψ."""
        ψ_hat = -ω_hat / ksq
        u_hat = 1j * ky * ψ_hat
        v_hat = -1j * kx * ψ_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        return u, v

    def nonlinear_term(ω_hat):
        """Compute Fourier transform of nonlinear advection term -u·∇ω."""
        u, v = velocity_from_vorticity(ω_hat)
        ω = np.real(np.fft.ifft2(ω_hat))
        ω_x = np.real(np.fft.ifft2(1j * kx * ω_hat))
        ω_y = np.real(np.fft.ifft2(1j * ky * ω_hat))
        adv = u * ω_x + v * ω_y
        return -np.fft.fft2(adv)

    def ns_rk4(w0, nu, dt, N_t, store_every, kx, ky, ksq, forced, forcing_amp, mask = None):
        """Time-integrate Navier–Stokes vorticity field with optional forcing."""
        w_hat = np.fft.fft2(w0)
        expLdt = np.exp(-nu * ksq * dt)
        expLdt2 = np.exp(-nu * ksq * dt / 2)

        y = np.linspace(0, 1, w0.shape[0], endpoint=False)
        Y = np.tile(y, (w0.shape[1], 1)).T

        # Random forcing amplitude for this trajectory
        if forced:
            forcing_freq = 4 * np.pi  # Kolmogorov forcing
        else:
            forcing_amp = 0.0
            forcing_freq = 0.0

        snapshots = [np.real(np.fft.ifft2(w_hat))]

        for n in range(1, N_t + 1):
            if forced:
                f = forcing_amp * (1.0 + 0.5 * np.sin(0.5 * n * dt)) * np.sin(forcing_freq * Y)
                f_hat = np.fft.fft2(f)
            else:
                f_hat = 0.0

            k1 = nonlinear_term(w_hat) + f_hat
            k2 = nonlinear_term(expLdt2 * (w_hat + 0.5 * dt * k1)) + f_hat
            k3 = nonlinear_term(expLdt2 * (w_hat + 0.5 * dt * k2)) + f_hat
            k4 = nonlinear_term(expLdt * (w_hat + dt * k3)) + f_hat

            w_hat = expLdt * w_hat + (dt / 6.0) * expLdt * (k1 + 2 * k2 + 2 * k3 + k4)

            if mask is not None:
                w_hat *= mask

            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft2(w_hat)))

        return np.stack(snapshots, axis=0)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    trajectories = []
    for j in range(num_samples):
        ω0 = random_vorticity_ic(N, kmax)
        ω_traj = ns_rk4(ω0, nu, dt, N_t, store_every, kx, ky, ksq, forced, forcing_amp, mask)
        trajectories.append(ω_traj.astype(dtype))
        print(
            f"Sample {j+1}/{num_samples}: stored {ω_traj.shape[0]} frames "
            f"(min={ω_traj.min():.3f}, max={ω_traj.max():.3f})"
        )

    # Pack data
    max_T = max(tr.shape[0] for tr in trajectories)
    ω_packed = np.zeros((num_samples, max_T, N, N), dtype=dtype)
    T_lengths = np.zeros(num_samples, dtype=np.int32)
    for j, tr in enumerate(trajectories):
        ω_packed[j, : tr.shape[0]] = tr
        T_lengths[j] = tr.shape[0]

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("omega", data=ω_packed, compression="gzip")
        f.create_dataset("X", data=x.astype(dtype))
        f.create_dataset("Y", data=y.astype(dtype))
        f.create_dataset("T_lengths", data=T_lengths)
        f.attrs["dt"] = dt
        f.attrs["store_every"] = store_every
        f.attrs["dt_eff"] = dt_eff
        f.attrs["nu"] = nu
        f.attrs["L"] = L
        f.attrs["L_t"] = L_t
        f.attrs["num_samples"] = num_samples
        f.attrs["forced"] = forced
        if spectral_filter_K0 is not None:
            f.attrs["spectral_filter_K0"] = spectral_filter_K0
        f.attrs["description"] = (
            "2D Navier–Stokes vorticity dataset: omega[sample, time, x, y], "
            + ("with sinusoidal forcing" if forced else "without forcing")
        )

    size_mb = os.path.getsize(save_path) / (1024**2)
    print(f"\nSaved dataset: {save_path} ({size_mb:.2f} MB)")
    print(f"ω shape: {ω_packed.shape}, X shape: {x.shape}, Y shape: {y.shape}")
    print(f"Total generation time: {timeit.default_timer() - t_start:.1f}s")
    print("==========================================\n")


if __name__ == "__main__":
    generate_navier_stokes_autoreg(
        N=128,
        L=1.0,
        L_t=5.0,
        dt=5e-3,
        nu=1e-5,
        num_samples=1000,
        kmax=12,
        store_every=1000,
        dtype=np.float32,
        save_dir="/scratch/mnhagen/datasets/navier_stokes",
        forced=True,
        spectral_filter_K0=16
    )
