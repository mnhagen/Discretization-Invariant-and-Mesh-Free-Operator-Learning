import numpy as np
import h5py
import os
import timeit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ================================================================
# Utility: anti-aliased Fourier-based downsampling
# ================================================================
def lowpass_and_decimate(u, factor):
    """
    Anti-aliased downsampling by integer factor in both dimensions.
    u: (N,N) array
    factor: int > 0

    Returns: (N//factor, N//factor)
    """
    if factor == 1:
        return u

    N = u.shape[0]
    cutoff = N // (2 * factor)

    # rFFT
    u_ft = np.fft.rfft2(u)
    H, W2 = u_ft.shape

    # Keep only the lowest <cutoff> modes in each dimension
    out_ft = np.zeros_like(u_ft)
    out_ft[:cutoff, :cutoff] = u_ft[:cutoff, :cutoff]
    out_ft[-cutoff:, :cutoff] = u_ft[-cutoff:, :cutoff]

    # Reconstruct and decimate
    u_filt = np.fft.irfft2(out_ft, s=(N, 2*(W2-1)))
    return u_filt[::factor, ::factor]


# ================================================================
# Main multires generator
# ================================================================
def generate_heat2d_multires(
    resolutions=[128, 64, 32],
    L=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=1e-3,
    num_samples=20,
    kmax=8,
    store_every=10,
    save_dir="/scratch/mnhagen/datasets/heat2d_multires/",
    dtype=np.float32,
):
    """
    Generate multiple heat-equation datasets at different resolutions.
    
    Key property:
    -------------------------------------------------------------
    A *single* initial condition is generated at the highest
    resolution only. Then it is *properly anti-aliased and 
    downsampled* to each lower resolution, and the PDE is solved
    separately at each resolution.
    -------------------------------------------------------------
    """

    # Sort resolutions high -> low
    resolutions = sorted(resolutions, reverse=True)
    N_high = resolutions[0]   # highest resolution
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Generating multires Heat2D datasets ===")
    print(f"Resolutions: {resolutions}")
    print(f"L={L}, L_t={L_t}, dt={dt}, store_every={store_every}")
    print(f"nu={nu}, num_samples={num_samples}\n")

    # ------------------------------------------------------------
    # Build spectral grid for highest resolution (for IC generation)
    # ------------------------------------------------------------
    dx_high = L / N_high
    kx_high = 2*np.pi*np.fft.fftfreq(N_high, d=dx_high)
    ky_high = 2*np.pi*np.fft.fftfreq(N_high, d=dx_high)
    kx_high, ky_high = np.meshgrid(kx_high, ky_high, indexing="ij")
    ksq_high = kx_high**2 + ky_high**2
    ksq_high[0, 0] = 1.0

    # ------------------------------------------------------------
    # Smooth random initial condition generator at highest res
    # ------------------------------------------------------------
    def random_ic_highres(N, kmax):
        coeffs = np.zeros((N, N), dtype=np.complex128)
        for i in range(1, kmax):
            for j in range(1, kmax):
                amp = np.exp(-0.5 * ((i**2 + j**2) / (kmax/2)**2))
                coeffs[i, j] = amp * (np.random.randn() + 1j*np.random.randn())
        coeffs[-kmax+1:, -kmax+1:] = np.conj(
            np.flip(np.flip(coeffs[1:kmax, 1:kmax], 0), 1)
        )
        u = np.real(np.fft.ifft2(coeffs))
        u /= np.max(np.abs(u))
        return u

    # ------------------------------------------------------------
    # Heat solver at arbitrary resolution N
    # ------------------------------------------------------------
    def heat_solver(u0, nu, dt, N_t, store_every, ksq):
        u_hat = np.fft.fft2(u0)
        expLdt = np.exp(-nu * ksq * dt)
        snapshots = [np.real(np.fft.ifft2(u_hat))]

        for n in range(1, N_t + 1):
            u_hat *= expLdt
            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft2(u_hat)))
        return np.stack(snapshots, axis=0)

    # ------------------------------------------------------------
    # Pre-generate all high-res ICs
    # ------------------------------------------------------------
    print("Generating high-resolution ICs...")
    ICs_high = [random_ic_highres(N_high, kmax) for _ in range(num_samples)]

    # ------------------------------------------------------------
    # Solve at each resolution and save output file
    # ------------------------------------------------------------
    for N in resolutions:
        print(f"\n=== Solving at resolution N={N} ===")

        # Build spectral grid for this resolution
        dx = L / N
        kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
        kx, ky = np.meshgrid(kx, ky, indexing="ij")
        ksq = kx**2 + ky**2
        ksq[0,0] = 1.0

        dt_eff = store_every * dt
        N_t = int(L_t / dt)

        # Solve trajectories
        trajectories = []
        for j in range(num_samples):
            factor = N_high // N
            u0 = lowpass_and_decimate(ICs_high[j], factor)
            traj = heat_solver(u0, nu, dt, N_t, store_every, ksq)
            trajectories.append(traj.astype(dtype))
            print(f"Sample {j+1}/{num_samples}: stored {traj.shape[0]} frames")

        # Pack and save
        max_T = max(tr.shape[0] for tr in trajectories)
        u_packed = np.zeros((num_samples, max_T, N, N), dtype=dtype)
        T_lengths = np.zeros(num_samples, dtype=np.int32)
        for j, tr in enumerate(trajectories):
            u_packed[j, :tr.shape[0]] = tr
            T_lengths[j] = tr.shape[0]

        # Save file
        filename = f"heat2D_autoreg_N{N}_multi.h5"
        path = os.path.join(save_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("u", data=u_packed, compression="gzip")
            f.create_dataset("X", data=(np.linspace(0, L, N, endpoint=False)).astype(dtype))
            f.create_dataset("Y", data=(np.linspace(0, L, N, endpoint=False)).astype(dtype))
            f.create_dataset("T_lengths", data=T_lengths)
            f.attrs["dt"] = dt
            f.attrs["store_every"] = store_every
            f.attrs["dt_eff"] = dt_eff
            f.attrs["nu"] = nu
            f.attrs["L"] = L
            f.attrs["L_t"] = L_t
            f.attrs["num_samples"] = num_samples
            f.attrs["description"] = (
                "2D Heat equation dataset: u[sample,time,x,y] "
                "generated from shared high-resolution ICs"
            )

        size_mb = os.path.getsize(path) / (1024**2)
        print(f"Saved: {path} ({size_mb:.2f} MB)")


# ================================================================
# Example usage
# ================================================================
if __name__ == "__main__":
    generate_heat2d_multires(
        resolutions=[512, 256, 128, 64, 32],   # pass any list of resolutions
        L=1.0,
        L_t=1.0,
        dt=1e-3,
        nu=1e-3,
        num_samples=1000,
        kmax=12,
        store_every=1000,
        dtype=np.float32,
        save_dir="/scratch/mnhagen/datasets/heat2d_multires/"
    )
