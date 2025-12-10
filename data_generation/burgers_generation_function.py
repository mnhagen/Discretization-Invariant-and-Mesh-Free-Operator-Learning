r"""
1D Burgers dataset generator using spectral RK4 solver.
Each sample: random initial condition -> integrated to final state.
Optionally saves one or more animations for sanity check.
Outputs an HDF5 dataset, compatible with FNO training code.
"""

import numpy as np
import h5py
import os
import timeit
import viz_tools


def generate_burgers(N_x, L_x, L_t, dt, nu, num_samples, kmax = 16, mu = 1.0,
                     save_dir = "/scratch/mnhagen/datasets/burgers/with_spatial/",
                     anim_dir = "/scratch/mnhagen/animations",
                     filename_params = [],
                     base_name = None, save_anim = False, anim_every = 200):
    """
    Callable function for generating 1D Burgers' equation data.

    N_x:                int, resolution/number of datapoints.
    L_x:                float, range of x-values. always starts at 0.
    L_t:                float, time between initial and final function.
    dt:                 float, time step size.  
    nu:                 float, Burgers' equation diffusion coefficient.
    num_samples:        int, number of samples to be generated.
    kmax:               int, number of Fourier modes to keep.
    mu:                 float, scaling parameter used for numerical solving.
    save_dir:           string, directory to which the generated datasets are saved.
    anim_dir:           string, directory to which animations are saved if save_anim is true.
    filename_params:    list of strings, parameters listed in the final name of the dataset with values.
    base_name:          string, base name to which the filename_params are appended.
    save_anim:          boolean, whether to save some animations, used for sanity checks.
    anim_every:         int, if save_anim is true, the interval at which animations are saved.
    
    Extensions to look into: variable inputs; ranges for L_t, nu etc. append in new columns. look into
    ordering; maybe align with FNO expectation, maybe just leave and keep permute
    """



    t1 = timeit.default_timer()

    # Parameters
    if filename is None:
        base_name = "burgers_1D"
        parts = []

        if filename_params:
            for p in filename_params:
                if isinstance(p, tuple):
                    key, val = p
                else:
                    key, val = p, locals().get(p, None)
                parts.append(f"{key}{val}")

        parts.append(f"Nx{N_x}")

        filename = base_name + "_" + "_".join(parts) + ".h5"

    save_path = os.path.join(save_dir, filename)
    dx = L_x/N_x
    X = np.linspace(0, L_x, N_x, endpoint=False)
    N_t = int(L_t / dt)


    # Helper: random smooth initial condition
    def random_initial_condition(N_x, kmax = N_x//4):
        """Create smooth random initial condition via truncated Fourier series."""
        coeffs = np.zeros(N_x, dtype=np.complex128)
        # Add some random low-frequency content
        coeffs[1:kmax] = (np.random.randn(kmax-1) + 1j*np.random.randn(kmax-1)) * np.exp(-0.5*(np.arange(1,kmax)/(kmax/4))**2)
        coeffs[-kmax+1:] = np.conj(np.flip(coeffs[1:kmax]))
        u = np.real(np.fft.ifft(coeffs))
        # Normalize amplitude
        u = u / np.max(np.abs(u))
        return u


    # Solver: pseudo-spectral RK4
    def burgers_rk4(u0, mu, nu, k, dt, N_t):
        u_hat = np.fft.fft(u0)
        N_x = len(u0)
        U_store = np.zeros((N_x, N_t))
        U_store[:, 0] = np.real(np.fft.ifft(u_hat))

        # === (1) Precompute integrating-factor ===
        expLdt = np.exp(-nu * (np.abs(k)**2) * dt)  # viscous decay over one full step
        expLdt2 = np.exp(-nu * (np.abs(k)**2) * dt / 2)  # half step

        def nonlinear(u_hat_local):
            u = np.real(np.fft.ifft(u_hat_local))
            u_x = np.real(np.fft.ifft(k * u_hat_local))
            return -mu * np.fft.fft(u * u_x)

        for n in range(1, N_t):
            # === (2) Integrate nonlinear part in integrating-factor form ===
            k1 = nonlinear(u_hat)
            k2 = nonlinear(expLdt2 * (u_hat + 0.5 * dt * k1))
            k3 = nonlinear(expLdt2 * (u_hat + 0.5 * dt * k2))
            k4 = nonlinear(expLdt * (u_hat + dt * k3))

            # === (3) Apply the integrating factor when updating ===
            u_hat = expLdt * u_hat + (dt / 6.0) * expLdt * (k1 + 2*k2 + 2*k3 + k4)

            if n % 10 == 0:
                U_store[:, n] = np.real(np.fft.ifft(u_hat))

        U_store[:, -1] = np.real(np.fft.ifft(u_hat))
        return U_store

    # Main generation loop
    k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx) * 1j

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"burgers_1D_{N_x}.h5")

    a = np.zeros((num_samples, N_x, 2))  # input: initial condition + coordinates
    u = np.zeros((num_samples, N_x))     # output: final field

    anim_idx = 1
    for j in range(num_samples):
        u0 = random_initial_condition(N_x, kmax = kmax)
        U = burgers_rk4(u0, mu, nu, k, dt, N_t)
        print("U min,max at end:", U[:, -1].min(), U[:, -1].max())
        print("Was last column filled? any nonzero:", np.any(U[:, -1] != 0))
        # Show a few intermediate columns:
        print("U stats at t indices 0, 10, 100, -1:")
        for idx in [0, 10, 100, -1]:
            if idx < U.shape[1]:
                col = U[:, idx]
                print(idx, np.min(col), np.max(col), np.mean(col), np.var(col))

        # store final snapshot
        u[j, :] = U[:, -1]
        # concatenate with grid info for FNO input
        grid = X.reshape(-1, 1)
        a[j, :, :] = np.concatenate([u0.reshape(-1, 1), grid], axis=1)

        print(f"Sample {j+1}/{num_samples} done.")

        # Optional sanity check animation
        if save_anim and (j % anim_every == 0):
            os.makedirs(anim_dir, exist_ok=True)
            viz_tools.anim_1D(X, U, dt, pas_d_images=50, save=True,
                            myxlim=(0, L_x), myylim=(-1, 1.5),
                            save_dir=anim_dir, filename=filename)
            anim_idx += 1


    # Save to HDF5
    with h5py.File(save_path, "w") as f:
        f.create_dataset("a", data=a)
        f.create_dataset("u", data=u)

    print(f"Saved dataset to {save_path}")
    print(f"Shape of a: {a.shape}, shape of u: {u.shape}")

    t2 = timeit.default_timer()
    print(f"Total generation time: {t2 - t1:.2f} seconds")
