import numpy as np
import h5py
import os
import timeit


# ================================================================
# Utility: anti-aliased Fourier-based downsampling (1D)
# ================================================================
def lowpass_and_decimate_1d(u, factor):
    """
    Anti-aliased downsampling by integer factor.

    u: (N,) array
    factor: int > 0

    Returns: (N//factor,) filtered + decimated 1D array
    """
    if factor == 1:
        return u

    N = u.shape[0]
    cutoff = N // (2 * factor)

    # rFFT: shape (N//2+1,)
    u_ft = np.fft.rfft(u)
    W2 = u_ft.shape[0]

    # Keep only low frequencies
    out_ft = np.zeros_like(u_ft)
    out_ft[:cutoff] = u_ft[:cutoff]

    # Inverse transform, specifying original length
    u_filt = np.fft.irfft(out_ft, n=N)

    return u_filt[::factor]


# ================================================================
# Main multires generator (1D)
# ================================================================
def generate_heat1d_multires(
    resolutions=[256, 128, 64, 32],
    L=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=1e-3,
    num_samples=20,
    kmax=8,
    store_every=10,
    save_dir="/scratch/mnhagen/datasets/heat1d_multires/",
    dtype=np.float32,
):
    """
    EXACT 1D analogue of generate_heat2d_multires.

    Key property:
    -------------------------------------------------------------
    A *single* initial condition is generated at the highest
    resolution only (via 1D Fourier random IC). Then it is
    anti-aliased and downsampled to each lower resolution,
    and the PDE is solved *separately* at each resolution.
    -------------------------------------------------------------
    """

    # Ensure decreasing order
    resolutions = sorted(resolutions, reverse=True)
    N_high = resolutions[0]
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Generating multires Heat1D datasets ===")
    print(f"Resolutions: {resolutions}")
    print(f"L={L}, L_t={L_t}, dt={dt}, store_every={store_every}")
    print(f"nu={nu}, num_samples={num_samples}\n")

    # ------------------------------------------------------------
    # Build spectral grid for highest resolution IC generation
    # ------------------------------------------------------------
    dx_high = L / N_high
    k_high = 2 * np.pi * np.fft.fftfreq(N_high, d=dx_high)
    ksq_high = k_high**2
    ksq_high[0] = 1.0

    # ------------------------------------------------------------
    # Smooth random initial condition generator (1D)
    # ------------------------------------------------------------
    def random_ic_highres_1d(N, kmax):
        coeffs = np.zeros(N, dtype=np.complex128)
        for k in range(1, kmax):
            amp = np.exp(-0.5 * (k**2) / (kmax / 2) ** 2)
            coeffs[k] = amp * (np.random.randn() + 1j*np.random.randn())
        coeffs[-kmax+1:] = np.conj(coeffs[1:kmax][::-1])

        u = np.real(np.fft.ifft(coeffs))
        u /= np.max(np.abs(u))
        return u

    # ------------------------------------------------------------
    # Heat solver in 1D (spectral exact integration)
    # ------------------------------------------------------------
    def heat_solver_1d(u0, nu, dt, N_t, store_every, ksq):
        u_hat = np.fft.fft(u0)
        expLdt = np.exp(-nu * ksq * dt)
        snapshots = [np.real(np.fft.ifft(u_hat))]

        for n in range(1, N_t + 1):
            u_hat *= expLdt
            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft(u_hat)))
        return np.stack(snapshots, axis=0)

    # ------------------------------------------------------------
    # Pre-generate high-res ICs
    # ------------------------------------------------------------
    print("Generating high-resolution ICs...")
    ICs_high = [random_ic_highres_1d(N_high, kmax) for _ in range(num_samples)]

    # ------------------------------------------------------------
    # Solve at each resolution and save
    # ------------------------------------------------------------
    for N in resolutions:
        print(f"\n=== Solving at resolution N={N} ===")

        dx = L / N
        k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        ksq = k**2
        ksq[0] = 1.0

        dt_eff = store_every * dt
        N_t = int(L_t / dt)

        trajectories = []
        for j in range(num_samples):
            factor = N_high // N
            u0 = lowpass_and_decimate_1d(ICs_high[j], factor)
            traj = heat_solver_1d(u0, nu, dt, N_t, store_every, ksq)
            trajectories.append(traj.astype(dtype))
            print(f"Sample {j+1}/{num_samples}: stored {traj.shape[0]} frames")

        # Pack into array
        max_T = max(tr.shape[0] for tr in trajectories)
        u_packed = np.zeros((num_samples, max_T, N), dtype=dtype)
        T_lengths = np.zeros(num_samples, dtype=np.int32)
        for j, tr in enumerate(trajectories):
            u_packed[j, :tr.shape[0], :] = tr
            T_lengths[j] = tr.shape[0]

        # Save
        filename = f"heat1D_autoreg_N{N}_multi.h5"
        path = os.path.join(save_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("u", data=u_packed, compression="gzip")
            f.create_dataset("X", data=(np.linspace(0, L, N, endpoint=False)).astype(dtype))
            f.create_dataset("T_lengths", data=T_lengths)
            f.attrs["dt"] = dt
            f.attrs["store_every"] = store_every
            f.attrs["dt_eff"] = dt_eff
            f.attrs["nu"] = nu
            f.attrs["L"] = L
            f.attrs["L_t"] = L_t
            f.attrs["num_samples"] = num_samples
            f.attrs["description"] = (
                "1D Heat equation dataset: u[sample,time,x] "
                "generated from shared high-resolution ICs"
            )

        size_mb = os.path.getsize(path) / (1024**2)
        print(f"Saved: {path} ({size_mb:.2f} MB)")


# ================================================================
# Example usage
# ================================================================
if __name__ == "__main__":
    generate_heat1d_multires(
        resolutions=[512, 256, 128, 64, 32],
        L=1.0,
        L_t=1.0,
        dt=1e-3,
        nu=1e-3,
        num_samples=1000,
        kmax=12,
        store_every=1000,
        dtype=np.float32,
        save_dir="/scratch/mnhagen/datasets/heat1d_multires/"
    )
