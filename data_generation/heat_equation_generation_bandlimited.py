import numpy as np
import h5py
import os
import timeit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def generate_heat2d_autoreg(
    N=64,
    L=1.0,
    L_t=1.0,
    dt=1e-3,
    nu=1e-3,
    num_samples=20,
    kmax=8,
    store_every=10,
    save_dir="/scratch/mnhagen/datasets/heat2d_autoreg/",
    filename=None,
    dtype=np.float32,
    make_animation=False,
    anim_frames=200,
    anim_fps=30,
    M_modes = 16,
):
    """
    Generate 2D heat-equation trajectories u_t = ν ∇²u on a periodic box.

    Parameters
    ----------
    N : int
        Grid resolution (N×N)
    L : float
        Domain length [0,L) in both x and y (periodic)
    L_t : float
        Total simulated time
    dt : float
        Time step
    nu : float
        Diffusivity
    num_samples : int
        Number of independent trajectories
    kmax : int
        Max Fourier frequency for random initial condition
    store_every : int
        Save every N steps (Δt_eff = store_every * dt)
    save_dir : str
        Directory to save dataset
    make_animation : bool
        Whether to save an animation of the first trajectory
    anim_frames : int
        Number of frames in the animation (independent of stored frames)
    anim_fps : int
        Frame rate of animation
    """

    t_start = timeit.default_timer()
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"heat2D_autoreg_N{N}_nu{nu}_samples{num_samples}_dt{dt}_store{store_every}_bandlimited{M_modes}.h5"
    save_path = os.path.join(save_dir, filename)

    # Spatial grids
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    ksq = kx**2 + ky**2
    ksq[0, 0] = 1.0  # avoid divide-by-zero
    N_t = int(L_t / dt)
    dt_eff = store_every * dt

    print(f"\n=== Heat equation 2D dataset ===")
    print(f"N={N}, L={L}, L_t={L_t}, dt={dt}, store_every={store_every} -> Δt_eff={dt_eff}")
    print(f"nu={nu}, num_samples={num_samples}")
    print(f"Saving to: {save_path}\n")

    # ------------------------------------------------------------------
    # Helper: smooth random initial condition
    # ------------------------------------------------------------------
    def random_ic_bandlimited(N, M):
        """
        Generate a strictly band-limited initial condition with
        frequencies |kx|, |ky| <= M.
        """
        coeffs = np.zeros((N, N), dtype=np.complex128)

        
        for i in range(-M, M+1):
            for j in range(-M, M+1):
                coeffs[i % N, j % N] = (np.random.randn() + 1j * np.random.randn())

        # Normalize
        u = np.real(np.fft.ifft2(coeffs))
        u /= np.max(np.abs(u))
        return u


    # ------------------------------------------------------------------
    # Time stepping (exact in Fourier space)
    # ------------------------------------------------------------------
    def heat_solver(u0, nu, dt, N_t, store_every, ksq, M = M_modes):
        """Integrate u_t = ν∇²u with explicit band-limiting every step."""
        u_hat = np.fft.fft2(u0)
        expLdt = np.exp(-nu * ksq * dt)

        snapshots = [np.real(np.fft.ifft2(u_hat))]

        for n in range(1, N_t + 1):
            u_hat *= expLdt

            # Band-limit: zero out all modes outside |k|<=M
            u_hat_bl = np.zeros_like(u_hat)
            u_hat_bl[:M+1, :M+1] = u_hat[:M+1, :M+1]
            u_hat_bl[-M:, :M+1] = u_hat[-M:, :M+1]
            u_hat_bl[:M+1, -M:] = u_hat[:M+1, -M:]
            u_hat_bl[-M:, -M:] = u_hat[-M:, -M:]
            u_hat = u_hat_bl

            if n % store_every == 0:
                snapshots.append(np.real(np.fft.ifft2(u_hat)))

        return np.stack(snapshots, axis=0)


    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    trajectories = []
    for j in range(num_samples):
        u0 = random_ic_bandlimited(N, kmax)
        u_traj = heat_solver(u0, nu, dt, N_t, store_every, ksq)
        trajectories.append(u_traj.astype(dtype))
        print(
            f"Sample {j+1}/{num_samples}: stored {u_traj.shape[0]} frames "
            f"(min={u_traj.min():.3f}, max={u_traj.max():.3f})"
        )

    # Pack & save
    max_T = max(tr.shape[0] for tr in trajectories)
    u_packed = np.zeros((num_samples, max_T, N, N), dtype=dtype)
    T_lengths = np.zeros(num_samples, dtype=np.int32)
    for j, tr in enumerate(trajectories):
        u_packed[j, : tr.shape[0]] = tr
        T_lengths[j] = tr.shape[0]

    with h5py.File(save_path, "w") as f:
        f.create_dataset("u", data=u_packed, compression="gzip")
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
        f.attrs["description"] = (
            "2D Heat equation dataset: u[sample,time,x,y]"
        )

    size_mb = os.path.getsize(save_path) / (1024**2)
    print(f"\nSaved dataset: {save_path} ({size_mb:.2f} MB)")
    print(f"u shape: {u_packed.shape}, X shape: {x.shape}, Y shape: {y.shape}")
    print(f"Total generation time: {timeit.default_timer() - t_start:.1f}s")
    print("==========================================\n")

    # ------------------------------------------------------------------
    # Optional animation
    # ------------------------------------------------------------------
    if make_animation:
        print("Generating animation ...")
        u0 = random_ic_bandlimited(N, kmax)
        u_hat = np.fft.fft2(u0)
        expLdt = np.exp(-nu * ksq * dt)
        N_anim_steps = int(L_t / dt)
        step_stride = max(1, N_anim_steps // anim_frames)
        frames = []
        for n in range(N_anim_steps):
            u_hat *= expLdt
            if n % step_stride == 0:
                frames.append(np.real(np.fft.ifft2(u_hat)))
        frames = np.stack(frames, axis=0)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(frames[0], cmap="inferno", origin="lower", extent=[0, L, 0, L])
        ax.set_title("2D Heat Equation")
        ax.set_xlabel("x"); ax.set_ylabel("y")

        def update(frame):
            im.set_data(frames[frame])
            ax.set_title(f"t = {frame * step_stride * dt:.3f} s")
            return [im]

        anim = FuncAnimation(fig, update, frames=frames.shape[0], interval=100, blit=True)

        anim_dir = "/scratch/mnhagen/animations/heat2d"
        os.makedirs(anim_dir, exist_ok=True)
        save_path_anim = os.path.join(anim_dir, f"heat2D_N{N}_nu{nu}_bandlimited{M_modes}.mp4")
        writer = FFMpegWriter(fps=anim_fps, bitrate=1800)
        anim.save(save_path_anim, writer=writer)
        print(f"Saved animation to: {save_path_anim}")


if __name__ == "__main__":
    generate_heat2d_autoreg(
        N=128,
        L=1.0,
        L_t=1.0,
        dt=1e-3,
        nu=1e-3,
        num_samples=1000,
        kmax=12,
        store_every=1000,
        dtype=np.float32,
        make_animation=True,
        anim_frames=200,
        anim_fps=30,
        save_dir="/scratch/mnhagen/datasets/heat2d",
        M_modes = 8
    )
