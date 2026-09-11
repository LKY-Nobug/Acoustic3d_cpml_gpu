"""Create an MP4 video from a 3-D acoustic wavefield stored on a CUDA GPU.

Expected array convention
-------------------------
wavefield_gpu[it, ix, iy, iz]

The function transfers only one 2-D slice per frame from GPU to CPU, so the
complete 4-D wavefield is never copied to host memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import Rectangle

Axis = Literal["x", "y", "z"]


def _validate_inputs(
    wavefield_gpu: cp.ndarray,
    dx: float,
    dt: float,
    axis: Axis,
    index: Optional[int],
    frame_step: int,
    n_pml: Optional[int],
) -> tuple[int, int, int, int, int]:
    if not isinstance(wavefield_gpu, cp.ndarray):
        raise TypeError("wavefield_gpu must be a CuPy array on the GPU.")
    if wavefield_gpu.ndim != 4:
        raise ValueError(
            "wavefield_gpu must have shape (nt, nx, ny, nz)."
        )
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive.")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    if int(frame_step) != frame_step or frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    nt, nx, ny, nz = (int(value) for value in wavefield_gpu.shape)
    axis_length = {"x": nx, "y": ny, "z": nz}[axis]
    slice_index = axis_length // 2 if index is None else int(index)
    if not 0 <= slice_index < axis_length:
        raise ValueError(
            f"index={slice_index} is outside axis '{axis}' of length "
            f"{axis_length}."
        )

    if n_pml is not None:
        if int(n_pml) != n_pml or n_pml < 0:
            raise ValueError("n_pml must be a non-negative integer or None.")
        n_pml = int(n_pml)
        if 2 * n_pml >= min(nx, ny, nz):
            raise ValueError("n_pml is too large for the wavefield dimensions.")

    return nt, nx, ny, nz, slice_index


def _extract_slice(
    wavefield_gpu: cp.ndarray,
    it: int,
    axis: Axis,
    index: int,
) -> cp.ndarray:
    """Return a 2-D GPU view ordered as [horizontal, vertical]."""
    if axis == "x":
        return wavefield_gpu[it, index, :, :]  # [y, z]
    if axis == "y":
        return wavefield_gpu[it, :, index, :]  # [x, z]
    return wavefield_gpu[it, :, :, index]  # [x, y]


def _plot_geometry(
    axis: Axis,
    shape: tuple[int, int, int],
    dx: float,
    n_pml: Optional[int],
) -> tuple[list[float], str, str]:
    nx, ny, nz = shape
    origin = -float(n_pml or 0) * dx

    if axis == "x":
        nh, nv = ny, nz
        xlabel, ylabel = "y (m)", "z (m)"
    elif axis == "y":
        nh, nv = nx, nz
        xlabel, ylabel = "x (m)", "z (m)"
    else:
        nh, nv = nx, ny
        xlabel, ylabel = "x (m)", "y (m)"

    extent = [
        origin,
        origin + (nh - 1) * dx,
        origin,
        origin + (nv - 1) * dx,
    ]
    return extent, xlabel, ylabel


def _draw_physical_boundary(
    ax: plt.Axes,
    extent: Sequence[float],
    dx: float,
    n_pml: Optional[int],
) -> None:
    if not n_pml:
        return

    offset = n_pml * dx
    x0 = extent[0] + offset
    y0 = extent[2] + offset
    width = (extent[1] - extent[0]) - 2.0 * offset
    height = (extent[3] - extent[2]) - 2.0 * offset
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
            label="physical model",
        )
    )
    ax.legend(loc="upper right")


def _estimate_vmax(
    wavefield_gpu: cp.ndarray,
    frame_indices: range,
    axis: Axis,
    index: int,
    percentile: float,
) -> float:
    """Estimate one fixed symmetric color limit without host-side copies."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100].")

    vmax = 0.0
    for it in frame_indices:
        frame = _extract_slice(wavefield_gpu, it, axis, index)
        value = float(cp.percentile(cp.abs(frame), percentile).item())
        vmax = max(vmax, value)

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(
            cp.max(cp.abs(_extract_slice(wavefield_gpu, 0, axis, index))).item()
        )
    return vmax if vmax > 0.0 else 1.0


def save_wavefield_video(
    wavefield_gpu: cp.ndarray,
    dx: float,
    dt: float,
    output_path: str | Path = "wavefield.mp4",
    *,
    axis: Axis = "z",
    index: Optional[int] = None,
    frame_step: int = 1,
    fps: int = 20,
    n_pml: Optional[int] = None,
    vmax: Optional[float] = None,
    percentile: float = 99.8,
    cmap: str = "seismic",
    dpi: int = 120,
    title: str = "3-D acoustic pressure wavefield",
    progress_every: int = 20,
) -> Path:
    """Write a pressure-wavefield slice sequence to an MP4 file.

    Parameters
    ----------
    wavefield_gpu
        CuPy array with shape ``(nt, nx, ny, nz)``. The array may include
        CPML cells.
    dx, dt
        Grid spacing in metres and time step in seconds.
    output_path
        Destination MP4 path.
    axis
        Normal direction of the displayed slice: ``'x'``, ``'y'`` or ``'z'``.
    index
        Slice index in the padded wavefield. ``None`` selects the centre.
    frame_step
        Write every ``frame_step``-th simulation time step.
    fps
        Output video frame rate.
    n_pml
        CPML width. When supplied, the physical-model boundary is drawn.
    vmax
        Fixed symmetric colour limit. If omitted, it is estimated from the
        selected frames using ``percentile``.
    percentile
        Robust percentile used to estimate ``vmax``. A fixed scale is used for
        all frames, making boundary absorption and amplitude decay comparable.
    cmap
        Matplotlib colour map.
    dpi
        Video resolution scale.
    title
        Figure title prefix.
    progress_every
        Print progress every N written frames. Set to 0 to disable.

    Returns
    -------
    pathlib.Path
        Path of the generated MP4 file.
    """
    nt, nx, ny, nz, slice_index = _validate_inputs(
        wavefield_gpu, dx, dt, axis, index, frame_step, n_pml
    )
    if int(fps) != fps or fps <= 0:
        raise ValueError("fps must be a positive integer.")
    if vmax is not None and (not np.isfinite(vmax) or vmax <= 0.0):
        raise ValueError("vmax must be a positive finite value or None.")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError(
            "FFmpeg was not found. Install ffmpeg and ensure it is available "
            "on the system PATH."
        )

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = range(0, nt, int(frame_step))
    if vmax is None:
        print("Estimating a fixed display amplitude...")
        vmax = _estimate_vmax(
            wavefield_gpu, frame_indices, axis, slice_index, percentile
        )

    extent, xlabel, ylabel = _plot_geometry(
        axis, (nx, ny, nz), dx, n_pml
    )
    first_gpu = _extract_slice(wavefield_gpu, 0, axis, slice_index)
    first = cp.asnumpy(first_gpu).T

    fig, ax = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)
    image = ax.imshow(
        first,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n{axis} index = {slice_index}, t = 0.000 ms"
    )
    _draw_physical_boundary(ax, extent, dx, n_pml)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("pressure amplitude")

    writer = FFMpegWriter(
        fps=int(fps),
        codec="libx264",
        metadata={"title": title},
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
    )

    total_frames = len(frame_indices)
    try:
        with writer.saving(fig, str(output_path), dpi=dpi):
            for frame_number, it in enumerate(frame_indices, start=1):
                frame = cp.asnumpy(
                    _extract_slice(wavefield_gpu, it, axis, slice_index)
                ).T
                image.set_data(frame)
                ax.set_title(
                    f"{title}\n{axis} index = {slice_index}, "
                    f"t = {it * dt * 1.0e3:.3f} ms"
                )
                writer.grab_frame()

                if progress_every and (
                    frame_number % progress_every == 0
                    or frame_number == total_frames
                ):
                    print(
                        f"Written {frame_number}/{total_frames} frames "
                        f"({100.0 * frame_number / total_frames:.1f}%)."
                    )
    finally:
        plt.close(fig)

    print(f"Video saved to: {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(
        "Import save_wavefield_video() and pass the wavefield_gpu returned "
        "by acoustic3d_cpml_gpu(). See the usage example below."
    )


# Minimal usage example
# ---------------------
# from acoustic3d_cpml_gpu import acoustic3d_cpml_gpu
# from wavefield_video_gpu import save_wavefield_video
#
# seismograms_gpu, wavefield_gpu = acoustic3d_cpml_gpu(
#     vp=vp,
#     dx=dx,
#     dt=dt,
#     nt=nt,
#     f0=f0,
#     source_xyz=source_xyz,
#     receiver_xyz=receiver_xyz,
#     n_pml=n_pml,
#     save_wavefield=True,
# )
#
# save_wavefield_video(
#     wavefield_gpu,
#     dx=dx,
#     dt=dt,
#     output_path="wavefield_z.mp4",
#     axis="z",
#     index=n_pml + vp.shape[2] // 2,
#     frame_step=2,
#     fps=20,
#     n_pml=n_pml,
# )
