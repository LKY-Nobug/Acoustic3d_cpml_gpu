"""3-D constant-density acoustic forward modeling on a staggered grid with CPML.

Array convention
----------------
vp[ix, iy, iz], where x is transverse, y is tunnel-heading direction,
and z is vertical. All coordinates and physical parameters use SI units.

The time stepping is GPU-only and uses CuPy float32 arrays.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import cupy as cp

ArrayLike = Union[np.ndarray, cp.ndarray]


def _cpml_coefficients(
    n: int,
    n_pml: int,
    dx: float,
    dt: float,
    vmax: float,
    f0: float,
    reflection: float = 1.0e-6,
    power: int = 3,
    kappa_max: float = 5.0,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return 1-D CPML arrays (a, b, kappa) for centers or faces."""
    i = cp.arange(n, dtype=cp.float32)
    width = cp.float32(n_pml)

    left = cp.clip((width - i) / width, 0.0, 1.0)
    right = cp.clip((i - cp.float32(n - n_pml - 1)) / width, 0.0, 1.0)
    eta = cp.maximum(left, right)
    eta_m = eta**power

    sigma_max = -(
        (power + 1) * vmax * np.log(reflection) / (2.0 * n_pml * dx)
    )
    sigma = cp.float32(sigma_max) * eta_m
    kappa = 1.0 + cp.float32(kappa_max - 1.0) * eta_m
    alpha = cp.where(eta > 0.0, cp.float32(np.pi * f0) * (1.0 - eta), 0.0)

    b = cp.exp(-(sigma / kappa + alpha) * cp.float32(dt))
    denominator = kappa * (sigma + kappa * alpha)
    denominator = cp.where(denominator > 0.0, denominator, 1.0)
    a = sigma * (b - 1.0) / denominator

    return (
        a.astype(cp.float32, copy=False),
        b.astype(cp.float32, copy=False),
        kappa.astype(cp.float32, copy=False),
    )


def _coordinate_to_index(
    xyz: Sequence[float],
    dx: float,
    shape: Tuple[int, int, int],
    n_pml: int,
    name: str,
) -> Tuple[int, int, int]:
    xyz_array = np.asarray(xyz, dtype=np.float64)
    if xyz_array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values: (x, y, z).")

    index = np.rint(xyz_array / dx).astype(np.int64)
    tolerance = max(1.0e-8, abs(dx) * 1.0e-6)
    if not np.allclose(index * dx, xyz_array, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must lie exactly on grid nodes with spacing dx={dx}.")

    if np.any(index < 0) or np.any(index >= np.asarray(shape)):
        raise ValueError(f"{name}={tuple(xyz_array)} lies outside the physical model.")

    index += n_pml
    return int(index[0]), int(index[1]), int(index[2])


def acoustic3d_cpml_gpu(
    vp: ArrayLike,
    dx: float,
    dt: float,
    nt: int,
    f0: float,
    snapshot_stride:int,
    source_xyz: Sequence[float],
    receiver_xyz: Sequence[Sequence[float]],
    n_pml: int,
    save_wavefield: bool = False,
) -> Tuple[cp.ndarray, Optional[cp.ndarray]]:
    """Run 3-D constant-density acoustic forward modeling on one CUDA GPU.

    Parameters
    ----------
    vp
        P-wave velocity model with shape ``(nx, ny, nz)`` in m/s.
    dx
        Uniform grid spacing in m.
    dt
        Time step in s. It is checked but never changed automatically.
    nt
        Number of time steps.
    f0
        Ricker-wavelet dominant frequency in Hz.
    source_xyz
        Source physical coordinate ``(x, y, z)`` in m.
    receiver_xyz
        Receiver coordinates with shape ``(n_receivers, 3)`` in m.
    n_pml
        CPML width in grid nodes, added outside all six model faces.
    save_wavefield
        If True, return pressure at every time step including CPML cells.

    Returns
    -------
    seismograms
        GPU array with shape ``(nt, n_receivers)`` containing pressure.
    wavefield
        GPU array with shape ``(nt, nx+2*n_pml, ny+2*n_pml,
        nz+2*n_pml)`` when enabled; otherwise None.
    """
    if dx <= 0.0 or dt <= 0.0 or f0 <= 0.0:
        raise ValueError("dx, dt and f0 must be positive.")
    if int(nt) != nt or nt <= 0:
        raise ValueError("nt must be a positive integer.")
    if int(n_pml) != n_pml or n_pml <= 0:
        raise ValueError("n_pml must be a positive integer.")
    nt, n_pml = int(nt), int(n_pml)

    vp_gpu = cp.asarray(vp, dtype=cp.float32)
    if vp_gpu.ndim != 3:
        raise ValueError("vp must be a 3-D array ordered as vp[nx, ny, nz].")
    if bool(cp.any(vp_gpu <= 0.0).item()):
        raise ValueError("All P-wave velocities must be positive.")

    physical_shape = tuple(int(n) for n in vp_gpu.shape)
    vmax = float(cp.max(vp_gpu).item())
    cfl = vmax * dt / dx
    if cfl >= 1.0 / np.sqrt(3.0):
        raise ValueError(
            f"Unstable time step: vmax*dt/dx={cfl:.4f} must be < 1/sqrt(3)."
        )

    source_index = _coordinate_to_index(
        source_xyz, dx, physical_shape, n_pml, "source_xyz"
    )

    receivers = np.asarray(receiver_xyz, dtype=np.float64)
    if receivers.ndim == 1:
        receivers = receivers.reshape(1, -1)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or receivers.shape[0] == 0:
        raise ValueError("receiver_xyz must have shape (n_receivers, 3).")
    receiver_indices = np.asarray(
        [
            _coordinate_to_index(row, dx, physical_shape, n_pml, "receiver_xyz")
            for row in receivers
        ],
        dtype=np.int32,
    )
    rx = cp.asarray(receiver_indices[:, 0])
    ry = cp.asarray(receiver_indices[:, 1])
    rz = cp.asarray(receiver_indices[:, 2])

    vp_gpu = cp.pad(vp_gpu, n_pml, mode="edge")
    bulk_modulus = vp_gpu * vp_gpu  # rho = 1, so K = rho*vp^2 = vp^2
    nx, ny, nz = (int(n) for n in vp_gpu.shape)

    # Pressure at cell centers; particle velocities at staggered faces.
    p = cp.zeros((nx, ny, nz), dtype=cp.float32)
    vx = cp.zeros((nx + 1, ny, nz), dtype=cp.float32)
    vy = cp.zeros((nx, ny + 1, nz), dtype=cp.float32)
    vz = cp.zeros((nx, ny, nz + 1), dtype=cp.float32)

    # CPML memory variables for pressure gradients at velocity locations.
    psi_px = cp.zeros((nx - 1, ny, nz), dtype=cp.float32)
    psi_py = cp.zeros((nx, ny - 1, nz), dtype=cp.float32)
    psi_pz = cp.zeros((nx, ny, nz - 1), dtype=cp.float32)

    # CPML memory variables for velocity divergence at pressure locations.
    psi_vx = cp.zeros_like(p)
    psi_vy = cp.zeros_like(p)
    psi_vz = cp.zeros_like(p)

    ax_c, bx_c, kx_c = _cpml_coefficients(nx, n_pml, dx, dt, vmax, f0)
    ay_c, by_c, ky_c = _cpml_coefficients(ny, n_pml, dx, dt, vmax, f0)
    az_c, bz_c, kz_c = _cpml_coefficients(nz, n_pml, dx, dt, vmax, f0)
    ax_f, bx_f, kx_f = _cpml_coefficients(nx + 1, n_pml, dx, dt, vmax, f0)
    ay_f, by_f, ky_f = _cpml_coefficients(ny + 1, n_pml, dx, dt, vmax, f0)
    az_f, bz_f, kz_f = _cpml_coefficients(nz + 1, n_pml, dx, dt, vmax, f0)

    # Broadcast-ready views.
    ax_c, bx_c, kx_c = ax_c[:, None, None], bx_c[:, None, None], kx_c[:, None, None]
    ay_c, by_c, ky_c = ay_c[None, :, None], by_c[None, :, None], ky_c[None, :, None]
    az_c, bz_c, kz_c = az_c[None, None, :], bz_c[None, None, :], kz_c[None, None, :]
    ax_f, bx_f, kx_f = (
        ax_f[1:-1, None, None],
        bx_f[1:-1, None, None],
        kx_f[1:-1, None, None],
    )
    ay_f, by_f, ky_f = (
        ay_f[None, 1:-1, None],
        by_f[None, 1:-1, None],
        ky_f[None, 1:-1, None],
    )
    az_f, bz_f, kz_f = (
        az_f[None, None, 1:-1],
        bz_f[None, None, 1:-1],
        kz_f[None, None, 1:-1],
    )

    time = cp.arange(nt, dtype=cp.float32) * cp.float32(dt)
    # t0 = cp.float32(1.5 / f0)
    t0 = cp.float32(1.0 / f0)
    argument = (cp.float32(np.pi * f0) * (time - t0)) ** 2
    source = (1.0 - 2.0 * argument) * cp.exp(-argument)

    seismograms = cp.empty((nt, receivers.shape[0]), dtype=cp.float32)
    # 保存方式:
    # "xyz" : 全三维波场
    # "xy"  : 固定z的水平剖面
    # "yz"  : 固定x的纵剖面

    save_mode = "yz"

    # 保存剖面位置（注意：是加CPML后的索引）
    slice_index = int(round(source_xyz[0] / dx)) + n_pml

    # save_mode = "xy"

    # # 保存剖面位置（注意：是加CPML后的索引）
    # slice_index = int(round(source_xyz[2] / dx)) + n_pml

    snapshot_steps = np.arange(
            0,
            nt,
            snapshot_stride
        )
    if save_wavefield:
        n_snapshots = len(snapshot_steps)

        if save_mode == "xyz":

            wavefield = cp.empty(
                (n_snapshots, nx, ny, nz),
                dtype=cp.float32
            )

        elif save_mode == "xy":

            # 固定 z
            wavefield = cp.empty(
                (n_snapshots, nx, ny),
                dtype=cp.float32
            )

        elif save_mode == "yz":

            # 固定 x
            wavefield = cp.empty(
                (n_snapshots, ny, nz),
                dtype=cp.float32
            )

    else:
        wavefield = None

    inv_dx = cp.float32(1.0 / dx)
    dt32 = cp.float32(dt)
    sx, sy, sz = source_index

    isnap = 0 #用于稀疏采样索引
    for it in range(nt):
        # Pressure gradients -> staggered particle velocities.
        dp = (p[1:, :, :] - p[:-1, :, :]) * inv_dx
        psi_px = bx_f * psi_px + ax_f * dp
        vx[1:-1, :, :] -= dt32 * (dp / kx_f + psi_px)

        dp = (p[:, 1:, :] - p[:, :-1, :]) * inv_dx
        psi_py = by_f * psi_py + ay_f * dp
        vy[:, 1:-1, :] -= dt32 * (dp / ky_f + psi_py)

        dp = (p[:, :, 1:] - p[:, :, :-1]) * inv_dx
        psi_pz = bz_f * psi_pz + az_f * dp
        vz[:, :, 1:-1] -= dt32 * (dp / kz_f + psi_pz)

        # Staggered velocity divergence -> pressure.
        dv = (vx[1:, :, :] - vx[:-1, :, :]) * inv_dx
        psi_vx = bx_c * psi_vx + ax_c * dv
        divergence = dv / kx_c + psi_vx

        dv = (vy[:, 1:, :] - vy[:, :-1, :]) * inv_dx
        psi_vy = by_c * psi_vy + ay_c * dv
        divergence += dv / ky_c + psi_vy

        dv = (vz[:, :, 1:] - vz[:, :, :-1]) * inv_dx
        psi_vz = bz_c * psi_vz + az_c * dv
        divergence += dv / kz_c + psi_vz

        p -= dt32 * bulk_modulus * divergence
        p[sx, sy, sz] += source[it]

        seismograms[it, :] = p[rx, ry, rz]
        if wavefield is not None:
            if it % snapshot_stride == 0:
                if save_mode == "xyz":

                    wavefield[isnap,:,:,:] = p


                elif save_mode == "xy":

                    # 保存 xy 平面
                    # 固定 z
                    wavefield[isnap,:,:] = p[:,:,slice_index]


                elif save_mode == "yz":

                    # 保存 yz 平面
                    # 固定 x
                    wavefield[isnap,:,:] = p[slice_index,:,:]
                isnap += 1

    return seismograms, wavefield, snapshot_steps



