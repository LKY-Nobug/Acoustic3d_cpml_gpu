"""
Video generator for 2-D wavefield slices saved from 3-D acoustic modeling.

Input:
    slice_wavefield_gpu[it, i1, i2]

Examples:
    yz slice: p[x_fixed, y, z]
    xy slice: p[x, y, z_fixed]
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cupy as cp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import Rectangle


def save_wavefield_video(
    slice_wavefield_gpu: cp.ndarray,
    snapshot_steps: Sequence[int],
    dx: float,
    dt: float,
    output_path: str | Path = "wavefield.mp4",
    *,
    slice_type: str = "yz",
    n_pml: int = 0,
    show_pml: bool = True,
    fps: int = 25,
    percentile: float = 99.8,
    cmap: str = "seismic",
    dpi: int = 120,
) -> Path:
    """
    Generate an MP4 video from saved 2-D GPU wavefield slices.

    Parameters
    ----------
    slice_wavefield_gpu
        CuPy array with shape:

        - yz slice: ``(nt_save, ny_padded, nz_padded)``
        - xy slice: ``(nt_save, nx_padded, ny_padded)``

        The spatial dimensions are assumed to contain CPML cells when
        ``n_pml > 0``.

    snapshot_steps
        Original forward-modeling time-step index corresponding to each
        saved frame.

    dx, dt
        Grid spacing in metres and time step in seconds.

    output_path
        Output MP4 file path.

    slice_type
        ``"yz"`` or ``"xy"``.

    n_pml
        CPML thickness in grid points on each side.

    show_pml
        If True, display the complete padded wavefield and draw the physical
        model boundary.

        If False, remove the CPML cells and display only the physical model.

    fps
        Video frame rate.

    percentile
        Absolute-amplitude percentile used to determine the dynamic colour
        limit for each frame.

    cmap
        Matplotlib colour map.

    dpi
        Output video resolution scale.

    Returns
    -------
    pathlib.Path
        Generated MP4 path.
    """

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if not isinstance(slice_wavefield_gpu, cp.ndarray):
        raise TypeError("slice_wavefield_gpu must be a CuPy array.")

    if slice_wavefield_gpu.ndim != 3:
        raise ValueError(
            "slice_wavefield_gpu must have shape (nt_save, n1, n2)."
        )

    if slice_type not in ("yz", "xy"):
        raise ValueError("slice_type must be 'yz' or 'xy'.")

    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive.")

    if int(n_pml) != n_pml or n_pml < 0:
        raise ValueError("n_pml must be a non-negative integer.")

    if int(fps) != fps or fps <= 0:
        raise ValueError("fps must be a positive integer.")

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100].")

    n_pml = int(n_pml)
    fps = int(fps)

    steps = np.asarray(snapshot_steps)

    if steps.ndim != 1:
        raise ValueError("snapshot_steps must be a one-dimensional sequence.")

    if len(steps) != slice_wavefield_gpu.shape[0]:
        raise ValueError(
            "snapshot_steps length must equal "
            "slice_wavefield_gpu.shape[0]."
        )

    if not writers.is_available("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not available. Install ffmpeg and ensure that it is "
            "on the system PATH."
        )

    nt_save, n1_padded, n2_padded = (
        int(value) for value in slice_wavefield_gpu.shape
    )

    if n_pml > 0 and (
        2 * n_pml >= n1_padded or 2 * n_pml >= n2_padded
    ):
        raise ValueError(
            "n_pml is too large for the dimensions of slice_wavefield_gpu."
        )

    # ------------------------------------------------------------
    # Select displayed spatial region
    # ------------------------------------------------------------
    if show_pml or n_pml == 0:
        # Keep the complete wavefield, including CPML.
        spatial_slice = (
            slice(None),
            slice(None),
        )

        display_n1 = n1_padded
        display_n2 = n2_padded

        origin = -n_pml * dx

        extent = [
            origin,
            origin + (display_n1 - 1) * dx,
            origin,
            origin + (display_n2 - 1) * dx,
        ]

    else:
        # Remove CPML from all four sides of the 2-D section.
        spatial_slice = (
            slice(n_pml, -n_pml),
            slice(n_pml, -n_pml),
        )

        display_n1 = n1_padded - 2 * n_pml
        display_n2 = n2_padded - 2 * n_pml

        # The cropped physical model starts at coordinate zero.
        extent = [
            0.0,
            (display_n1 - 1) * dx,
            0.0,
            (display_n2 - 1) * dx,
        ]

    # ------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------
    if slice_type == "yz":
        # Stored frame is [y, z], then transposed for imshow:
        # horizontal axis -> y, vertical axis -> z.
        xlabel = "y (m)"
        ylabel = "z (m)"
    else:
        # Stored frame is [x, y], then transposed for imshow:
        # horizontal axis -> x, vertical axis -> y.
        xlabel = "x (m)"
        ylabel = "y (m)"

    # ------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # First frame
    # ------------------------------------------------------------
    first_gpu = slice_wavefield_gpu[
        0,
        spatial_slice[0],
        spatial_slice[1],
    ]

    first = cp.asnumpy(first_gpu).T

    amp = float(
        cp.percentile(
            cp.abs(first_gpu),
            percentile,
        ).item()
    )

    if not np.isfinite(amp) or amp <= 0.0:
        amp = 1.0

    # ------------------------------------------------------------
    # Plot initialization
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8, 6),
        constrained_layout=True,
    )

    img = ax.imshow(
        first,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-amp,
        vmax=amp,
        interpolation="nearest",
        aspect="equal",
    )

    # Only draw the physical-region boundary when CPML is displayed.
    if show_pml and n_pml > 0:
        physical_origin = 0.0

        physical_width = (
            n1_padded - 2 * n_pml - 1
        ) * dx

        physical_height = (
            n2_padded - 2 * n_pml - 1
        ) * dx

        ax.add_patch(
            Rectangle(
                (physical_origin, physical_origin),
                physical_width,
                physical_height,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                edgecolor="black",
                label="physical model",
            )
        )

        ax.legend(loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("pressure")

    region_name = (
        "physical model + CPML"
        if show_pml
        else "physical model"
    )

    # ------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",
        ],
    )

    with writer.saving(
        fig,
        str(output_path),
        dpi=dpi,
    ):
        for i in range(nt_save):
            frame_gpu = slice_wavefield_gpu[
                i,
                spatial_slice[0],
                spatial_slice[1],
            ]

            # Dynamic colour scale computed only from the displayed region.
            amp = float(
                cp.percentile(
                    cp.abs(frame_gpu),
                    percentile,
                ).item()
            )

            if not np.isfinite(amp) or amp <= 0.0:
                amp = 1.0

            frame = cp.asnumpy(frame_gpu).T

            img.set_data(frame)
            img.set_clim(-amp, amp)
            cbar.update_normal(img)

            ax.set_title(
                f"{slice_type} wavefield — {region_name}\n"
                f"t = {steps[i] * dt * 1000.0:.3f} ms"
            )

            writer.grab_frame()

    plt.close(fig)

    print(f"saved: {output_path.resolve()}")

    return output_path


# if __name__ == "__main__":
#     print(
#         "Import save_wavefield_video() and pass the saved "
#         "3-D CuPy slice history."
#     )