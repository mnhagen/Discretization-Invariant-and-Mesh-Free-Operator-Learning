import numpy as np
import h5py
import os
import timeit


def spectral_downsample(omega_hi, N_target):
    """
    Downsample omega_hi (shape HxH) to NxN using spectral truncation.
    Assumes input is a real field.
    """
    H = omega_hi.shape[0]
    assert H >= N_target

    # FFT of high-res
    om_hat = np.fft.fft2(omega_hi)

    # Compute low-pass window
    N_half = N_target // 2
    out_hat = np.zeros((N_target, N_target), dtype=np.complex128)

    # Copy lowest frequencies into center block
    out_hat[:N_half, :N_half] = om_hat[:N_half, :N_half]
    out_hat[-N_half:, :N_half] = om_hat[-N_half:, :N_half]

    # iFFT to target resolution
    return np.real(np.fft.ifft2(out_hat))


def generate_navier_stokes_multires(
    resolutions=[128, 64, 32],
    L=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=1e-3,
    num_samples=20,
    kmax=8,
    store_every=10,
    save_dir="/scratch/mnhagen/datasets/navier_stokes_multires/",
    dtype=np.float32,
    forced=True,
    forcing_amp=0.5,
    spectral_filter_K0=None,
):
    """
    Generate Navier–Stokes trajectories at multiple resolutions,
    all sharing the SAME initial conditions (derived from highest-res IC
    via spectral downsampling).

    One HDF5 dataset is saved per resolution.
    """

    t_start = timeit.default_timer()
    os.makedirs(save_dir, exist_ok=True)

    resolutions = sorted(resolutions)[::-1]  # highest → lowest
    N_max = resolutions[0]

    print("\n=== Generating multires Navier–Stokes datasets ===")
    print(f"Resolutions: {resolutions}")
    print(f"L={L}, L_t={L_t}, dt={dt}, store_every={store_every} → Δt_eff={store_every*dt}")
    print(f"nu={nu}, forced={forced}, forcing_amp={forcing_amp}")
    print(f"Shared ICs from highest resolution N={N_max}\n")

    # -------------------------------------------------------------------------
    # Helper: random smooth initial vorticity (high-res only)
    # -------------------------------------------------------------------------
    def random_vorticity_ic(N, kmax):
        coeffs = np.zeros((N, N), dtype=np.complex128)
        for i in range(1, kmax):
            for j in range(1, kmax):
                amp = np.exp(-0.5 * ((i*i + j*j) / (kmax/2)**2))
                coeffs[i,j] = amp * (np.random.randn() + 1j*np.random.randn())
        coeffs[-kmax+1:, -kmax+1:] = np.conj(np.flip(np.flip(coeffs[1:kmax, 1:kmax],0),1))
        w = np.real(np.fft.ifft2(coeffs))
        w /= np.max(np.abs(w)) + 1e-12
        return w

    # -------------------------------------------------------------------------
    # Per-resolution solver setup (each grid needs its own kx, ky, Laplacian)
    # -------------------------------------------------------------------------
    def precompute_grid(N):
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        kx = 2*np.pi * np.fft.fftfreq(N, d=dx)
        ky = 2*np.pi * np.fft.fftfreq(N, d=dx)
        kx, ky = np.meshgrid(kx, ky, indexing="ij")
        ksq = kx**2 + ky**2
        ksq[0,0] = 1.0
        return dx, x, y, kx, ky, ksq

    # Solver step (same as yours)
    def velocity_from_vorticity(ω_hat, kx, ky, ksq):
        ψ_hat = -ω_hat / ksq
        u_hat = 1j * ky * ψ_hat
        v_hat = -1j * kx * ψ_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        return u, v

    def nonlinear_rhs(ω_hat, kx, ky, ksq):
        u, v = velocity_from_vorticity(ω_hat, kx, ky, ksq)
        ω = np.real(np.fft.ifft2(ω_hat))
        ω_x = np.real(np.fft.ifft2(1j*kx*ω_hat))
        ω_y = np.real(np.fft.ifft2(1j*ky*ω_hat))
        adv = u*ω_x + v*ω_y
        return -np.fft.fft2(adv)

    def integrate_NS(w0, N, kx, ky, ksq, nu, dt, N_t, store_every, forced, forcing_amp, mask):
        w_hat = np.fft.fft2(w0)
        expL = np.exp(-nu*ksq*dt)
        expL_half = np.exp(-nu*ksq*(dt/2))

        # Precompute forcing coordinates
        ygrid = np.linspace(0,1,N,endpoint=False)
        Y = np.tile(ygrid, (N,1)).T

        forcing_freq = 4*np.pi if forced else 0.0
        forcing_amp_used = forcing_amp if forced else 0.0

        snapshots = [np.real(np.fft.ifft2(w_hat))]

        for n in range(1, N_t+1):
            if forced:
                f = forcing_amp_used * (1.0 + 0.5*np.sin(0.5*n*dt)) * np.sin(forcing_freq*Y)
                f_hat = np.fft.fft2(f)
            else:
                f_hat = 0.0

            k1 = nonlinear_rhs(w_hat, kx, ky, ksq) + f_hat
            k2 = nonlinear_rhs(expL_half*(w_hat + 0.5*dt*k1), kx, ky, ksq) + f_hat
            k3 = nonlinear_rhs(expL_half*(w_hat + 0.5*dt*k2), kx, ky, ksq) + f_hat
            k4 = nonlinear_rhs(expL*(w_hat + dt*k3), kx, ky, ksq) + f_hat

            w_hat = expL*w_hat + (dt/6.0)*expL*(k1 + 2*k2 + 2*k3 + k4)

            if mask is not None:
                w_hat *= mask

            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft2(w_hat)))

        return np.stack(snapshots, axis=0)

    # -------------------------------------------------------------------------
    # Generate high-res initial conditions
    # -------------------------------------------------------------------------
    print("Generating high-resolution ICs...")
    ICs_hi = [random_vorticity_ic(N_max, kmax) for _ in range(num_samples)]

    # -------------------------------------------------------------------------
    # Loop over resolutions
    # -------------------------------------------------------------------------
    for N in resolutions:
        print(f"\n=== Solving at resolution N={N} ===")

        # Precompute operators for resolution N
        dx, X, Y, kx, ky, ksq = precompute_grid(N)
        N_t = int(L_t/dt)

        # Optional spectral filter
        if spectral_filter_K0 is not None:
            mask = np.zeros((N,N), dtype=float)
            K0 = spectral_filter_K0
            mask[:K0,:K0] = 1.0
            mask[-K0:,:K0] = 1.0
        else:
            mask = None

        trajectories = []

        for j in range(num_samples):
            # Downsample IC
            if N == N_max:
                w0 = ICs_hi[j]
            else:
                w0 = spectral_downsample(ICs_hi[j], N)

            # Integrate PDE on this resolution
            traj = integrate_NS(
                w0, N, kx, ky, ksq, nu, dt, N_t, store_every,
                forced, forcing_amp, mask
            )
            trajectories.append(traj.astype(dtype))

            print(f"Sample {j+1}/{num_samples}, frames={traj.shape[0]}")

        # Pack & save
        max_T = max(t.shape[0] for t in trajectories)
        W = np.zeros((num_samples, max_T, N, N), dtype=dtype)
        T_lengths = np.zeros(num_samples, dtype=np.int32)

        for j, tr in enumerate(trajectories):
            W[j, :tr.shape[0]] = tr
            T_lengths[j] = tr.shape[0]

        save_path = os.path.join(
            save_dir,
            f"navier_stokes2D_autoreg_N{N}_multi.h5"
        )
        with h5py.File(save_path, "w") as f:
            f.create_dataset("omega", data=W, compression="gzip")
            f.create_dataset("X", data=X.astype(dtype))
            f.create_dataset("Y", data=Y.astype(dtype))
            f.create_dataset("T_lengths", data=T_lengths)
            f.attrs["dt"] = dt
            f.attrs["store_every"] = store_every
            f.attrs["dt_eff"] = dt * store_every
            f.attrs["nu"] = nu
            f.attrs["L"] = L
            f.attrs["L_t"] = L_t
            f.attrs["num_samples"] = num_samples
            f.attrs["forced"] = forced
            f.attrs["description"] = (
                f"Navier–Stokes vorticity dataset at N={N}, "
                "sharing ICs across resolutions via spectral truncation."
            )

        size_mb = os.path.getsize(save_path) / (1024**2)
        print(f"Saved {save_path} ({size_mb:.2f} MB)")

    print(f"\nTotal time: {timeit.default_timer() - t_start:.1f}s")
    print("========================================\n")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    generate_navier_stokes_multires(
        resolutions=[256, 128, 64, 32],
        L=1.0,
        L_t=5.0,
        dt=5e-3,
        nu=1e-5,
        num_samples=100,
        kmax=12,
        store_every=1000,
        dtype=np.float32,
        save_dir="/scratch/mnhagen/datasets/navier_stokes_multires",
        forced=True,
        forcing_amp=0.5,
        spectral_filter_K0=16,
    )
