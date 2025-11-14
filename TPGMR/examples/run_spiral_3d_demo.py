#!/usr/bin/env python3
"""Train TP-GMR on synthetic 3D spiral demos and plot reproductions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import sys

import matplotlib
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tpgmm_pylib.TPGMR import Demonstration, Frame, demos_from_npz, run_tpgmr_from_demos
from tpgmm_pylib.TPGMR.examples.generate_spiral_3d_demos import (
    DEFAULT_OUTPUT,
    generate_spiral_dataset,
)


_PYLAB = None


def get_pyplot(interactive: bool):
    global _PYLAB
    if _PYLAB is not None:
        return _PYLAB

    if not interactive:
        matplotlib.use("Agg")
    else:
        current = matplotlib.get_backend().lower()
        if "agg" in current:
            for candidate in ("Qt5Agg", "TkAgg", "MacOSX"):
                try:
                    matplotlib.use(candidate)
                    break
                except Exception:
                    continue

    import matplotlib.pyplot as plt  # type: ignore

    _PYLAB = plt
    return plt


def ensure_dataset(path: Path, regen: bool, seed: int) -> Path:
    if regen or not path.exists():
        return generate_spiral_dataset(path, seed=seed)
    return path


def arrow_from_frame(ax, frame: Frame, length: float = 0.12):
    origin = frame.b[1:]
    axes = frame.A[1:, 1:] @ (np.eye(3) * length)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for idx in range(3):
        vec = axes[:, idx]
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=colors[idx],
            linewidth=1.2,
            arrow_length_ratio=0.2,
        )


def ellipsoid_mesh(center: np.ndarray, cov: np.ndarray, n_phi: int = 24, n_theta: int = 12):
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones_like(phi), np.cos(theta))
    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-9)
    radii = np.sqrt(eigvals)
    transform = eigvecs @ np.diag(radii)
    ellip = transform @ sphere
    ellip = ellip + center[:, None]
    X = ellip[0].reshape(n_phi, n_theta)
    Y = ellip[1].reshape(n_phi, n_theta)
    Z = ellip[2].reshape(n_phi, n_theta)
    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            verts = [
                [X[i, j], Y[i, j], Z[i, j]],
                [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]],
                [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]],
                [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]],
            ]
            faces.append(verts)
    return faces


def plot_component_ellipsoids(ax, reproduction, color: np.ndarray, alpha: float = 0.12):
    if reproduction.mu_components is None or reproduction.sigma_components is None:
        return
    mu = reproduction.mu_components[1:, :]
    sigma = reproduction.sigma_components[1:, 1:, :]
    nb_components = mu.shape[1]
    for idx in range(nb_components):
        faces = ellipsoid_mesh(mu[:, idx], sigma[:, :, idx])
        collection = Poly3DCollection(faces, facecolor=color, alpha=alpha, linewidths=0.0)
        ax.add_collection3d(collection)


def scatter_keypoints(ax, traj: np.ndarray, color: np.ndarray):
    ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0], marker="*", color=color, s=40)
    ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], marker="o", color=color, s=30)


def plot_results(
    demos_xyz: np.ndarray,
    reproductions,
    plot_path: Path,
    interactive: bool,
):
    plt = get_pyplot(interactive)
    nb_samples = len(demos_xyz)
    repro_train = reproductions[:nb_samples]
    repro_new = reproductions[nb_samples:]

    fig = plt.figure(figsize=(13, 6))
    ax_train = fig.add_subplot(121, projection="3d")
    ax_new = fig.add_subplot(122, projection="3d")
    axes = [ax_train, ax_new]
    colors = plt.cm.tab10(np.linspace(0, 1, max(nb_samples, len(repro_new) or 1)))

    for idx, demo in enumerate(demos_xyz):
        color = colors[idx % len(colors)]
        ax_train.plot(demo[0], demo[1], demo[2], "--", color=color, alpha=0.5)
        scatter_keypoints(ax_train, demo, color)

    for idx, rep in enumerate(repro_train):
        color = colors[idx % len(colors)]
        traj = rep.data
        ax_train.plot(traj[0], traj[1], traj[2], color=color, linewidth=2.0)
        scatter_keypoints(ax_train, traj, color)
        arrow_from_frame(ax_train, rep.frames[0], length=0.1)
        arrow_from_frame(ax_train, rep.frames[-1], length=0.1)
        plot_component_ellipsoids(ax_train, rep, color=color, alpha=0.14)

    for idx, rep in enumerate(repro_new):
        color = colors[idx % len(colors)]
        traj = rep.data
        ax_new.plot(traj[0], traj[1], traj[2], color=color, linewidth=2.0)
        scatter_keypoints(ax_new, traj, color)
        arrow_from_frame(ax_new, rep.frames[0], length=0.12)
        arrow_from_frame(ax_new, rep.frames[-1], length=0.12)
        plot_component_ellipsoids(ax_new, rep, color=color, alpha=0.18)

    for ax, title in zip(axes, ["Training contexts", "Interpolated contexts"]):
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.grid(True, alpha=0.2)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])

    legend_handles = [
        Line2D([], [], linestyle="--", color="0.4", label="Demo"),
        Line2D([], [], linestyle="-", color="k", label="Reproduction"),
        Line2D([], [], linestyle="None", marker="*", color="k", label="Start"),
        Line2D([], [], linestyle="None", marker="o", color="k", label="Goal"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.suptitle("3D spiral TP-GMR reproductions")
    plot_path = plot_path.resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=170)
    if interactive:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved spiral plot to {plot_path}")


def run_pipeline(
    demos: Sequence[Demonstration],
    nb_states: int,
    nb_new_contexts: int,
    diag_reg: float,
    seed: int,
):
    return run_tpgmr_from_demos(
        demos,
        nb_states=nb_states,
        nb_repros=nb_new_contexts,
        diag_reg_factor=diag_reg,
        random_state=seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="NPZ dataset path (auto-generated if missing).",
    )
    parser.add_argument("--regen-dataset", action="store_true", help="Force regeneration of the dataset.")
    parser.add_argument("--nb-states", type=int, default=4, help="Number of GMM components.")
    parser.add_argument("--diag-reg", type=float, default=1e-4, help="Covariance regularization.")
    parser.add_argument("--nb-new-contexts", type=int, default=4, help="New contexts to synthesize.")
    parser.add_argument("--seed", type=int, default=9, help="Random seed for context synthesis.")
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("tpgmm_pylib/TPGMR/examples/spiral_3d_results.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--interactive", action="store_true", help="Display the plot with an interactive backend.")
    args = parser.parse_args(argv)

    dataset_path = ensure_dataset(args.dataset, args.regen_dataset, seed=args.seed)
    demos, metadata = demos_from_npz(dataset_path)
    result = run_pipeline(
        demos,
        nb_states=args.nb_states,
        nb_new_contexts=args.nb_new_contexts,
        diag_reg=args.diag_reg,
        seed=args.seed,
    )

    with np.load(dataset_path, allow_pickle=False) as data:
        demos_xyz = np.array(data["trajectories"])

    plot_results(demos_xyz, result["reproductions"], args.plot_out, args.interactive)


if __name__ == "__main__":
    main()
