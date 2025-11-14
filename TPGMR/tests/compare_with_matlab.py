#!/usr/bin/env python3
"""Compare the standalone TP-GMR port with MATLAB and generate the reference plot."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tpgmm_pylib.TPGMR import Frame, model_to_numpy_dict, run_tpgmr_demo

FloatArray = NDArray[np.float64]
COL_PEGS = np.array(
    [[0.2863, 0.0392, 0.2392], [0.9137, 0.4980, 0.0078]], dtype=float
)
PEG_MESH = np.array(
    [
        [-4, -4, -1.5, -1.5, 1.5, 1.5, 4, 4, -4],
        [-3.5, 10, 10, -1, -1, 10, 10, -3.5, -3.5],
    ],
    dtype=float,
) * 1e-1
LIM_AXES = (-1.2, 0.8, -1.1, 0.9)


def load_matlab_reference(mat_path: Path) -> dict:
    return loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def contexts_from_matlab_struct(r_struct) -> List[List[Frame]]:
    contexts: List[List[Frame]] = []
    for entry in np.atleast_1d(r_struct):
        frames = []
        for frame in np.atleast_1d(entry.p):
            frames.append(
                Frame(
                    A=np.array(frame.A, dtype=float),
                    b=np.array(frame.b, dtype=float).reshape(-1),
                )
            )
        contexts.append(frames)
    return contexts


def max_abs_diff(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def compare_stage(name: str, arr_py, arr_mat, tol: float) -> Tuple[bool, float]:
    diff = max_abs_diff(arr_py, arr_mat)
    return diff <= tol, diff


def verify_against_matlab(
    data_path: Path, mat_reference: Path, tol: float
) -> Tuple[bool, list, dict, dict, List[List[Frame]], np.ndarray]:
    mat = load_matlab_reference(mat_reference)
    nb_samples = int(mat["nbSamples"])
    contexts = contexts_from_matlab_struct(mat["r"])
    nb_repros = len(contexts) - nb_samples
    py_result = run_tpgmr_demo(
        data_path,
        contexts_override=contexts,
        nb_repros=nb_repros,
    )
    model_np = model_to_numpy_dict(py_result["model"])
    repro_data = np.stack([rep.data for rep in py_result["reproductions"]])
    repro_sigma = np.stack([rep.sigma for rep in py_result["reproductions"]])
    mat_model = mat["model"]
    mat_repro_data = np.stack(
        [np.array(entry.Data, dtype=float) for entry in np.atleast_1d(mat["r"])]
    )
    mat_repro_sigma = np.stack(
        [np.array(entry.Sigma, dtype=float) for entry in np.atleast_1d(mat["r"])]
    )

    stages = [
        ("Tensor Data", py_result["Data"], mat["Data"]),
        ("DataIn", py_result["DataIn"], mat["DataIn"]),
        (
            "Model Priors",
            model_np["Priors"],
            np.array(mat_model.Priors, dtype=float).ravel(),
        ),
        ("Model Mu", model_np["Mu"], np.array(mat_model.Mu, dtype=float)),
        ("Model Sigma", model_np["Sigma"], np.array(mat_model.Sigma, dtype=float)),
        ("MuGMR", py_result["MuGMR"], mat["MuGMR"]),
        ("SigmaGMR", py_result["SigmaGMR"], mat["SigmaGMR"]),
        ("Reproduction Data", repro_data, mat_repro_data),
        ("Reproduction Sigma", repro_sigma, mat_repro_sigma),
    ]
    report = []
    overall_ok = True
    for name, arr_py, arr_mat in stages:
        ok, diff = compare_stage(name, arr_py, arr_mat, tol)
        overall_ok &= ok
        report.append((name, diff, tol, ok))
    demos_struct = np.atleast_1d(mat["s"])
    return overall_ok, report, py_result, mat, contexts, demos_struct


def plot_pegs(ax, frames: Sequence[Frame]):
    for frame, color in zip(frames, COL_PEGS):
        disp = frame.A[1:, 1:] @ PEG_MESH + frame.b[1:, None]
        ax.fill(disp[0], disp[1], color=color, alpha=0.6, linewidth=0)


def plot_gaussians(ax, data: FloatArray, sigma: FloatArray, color: FloatArray, step: int = 5):
    for idx in range(0, data.shape[1], step):
        cov = sigma[:, :, idx]
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)
        ell = Ellipse(
            xy=data[:, idx],
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.25,
        )
        ax.add_patch(ell)


def format_axes(ax, title: str):
    ax.set_xlim(LIM_AXES[0], LIM_AXES[1])
    ax.set_ylim(LIM_AXES[2], LIM_AXES[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(False)


def render_plot(
    demos_struct,
    contexts: Sequence[Sequence[Frame]],
    reproductions,
    nb_samples: int,
    nb_repros: int,
    out_path: Path,
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    clrmap = plt.get_cmap("jet")
    colors = clrmap(np.linspace(0, 0.95, nb_samples))

    ax = axs[0]
    for n, demo in enumerate(demos_struct):
        frames = contexts[n]
        plot_pegs(ax, frames)
        data = np.array(demo.Data, dtype=float)
        ax.plot(data[1, 0], data[2, 0], "o", markersize=4, color=colors[n])
        ax.plot(data[1, :], data[2, :], linewidth=1.5, color=colors[n])
    format_axes(ax, "Demonstrations")

    ax = axs[1]
    for n in range(nb_samples):
        plot_pegs(ax, reproductions[n].frames)
    for n in range(nb_samples):
        traj = reproductions[n].data
        covs = reproductions[n].sigma
        plot_gaussians(ax, traj, covs, colors[n])
        ax.plot(traj[0, 0], traj[1, 0], "o", markersize=4, color=colors[n])
        ax.plot(traj[0, :], traj[1, :], linewidth=1.5, color=colors[n])
    format_axes(ax, "Reproductions with GMR")

    ax = axs[2]
    grey = np.array([0.2, 0.2, 0.2])
    for rep in reproductions[nb_samples : nb_samples + nb_repros]:
        plot_pegs(ax, rep.frames)
    for rep in reproductions[nb_samples : nb_samples + nb_repros]:
        traj = rep.data
        covs = rep.sigma
        plot_gaussians(ax, traj, covs, grey)
        ax.plot(traj[0, 0], traj[1, 0], "o", markersize=4, color=grey)
        ax.plot(traj[0, :], traj[1, :], linewidth=1.5, color=grey)
    format_axes(ax, "New reproductions with GMR")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("demos/data/Data02.mat"),
        help="Path to the PbDlib Data02.mat file.",
    )
    parser.add_argument(
        "--matlab-reference",
        type=Path,
        default=Path("pbd_tpgmm_pylib/matlab_outputs.mat"),
        help="Path to the MATLAB baseline exported from demo_TPGMR01.m",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Maximum absolute deviation allowed for all comparison stages.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("tpgmm_pylib/TPGMR/tests/python_tpgmr_demo.png"),
        help="Where to save the reproduction plot (Agg backend).",
    )
    args = parser.parse_args()
    data_path = (REPO_ROOT / args.data_path).resolve()
    matlab_reference = (REPO_ROOT / args.matlab_reference).resolve()
    ok, report, py_result, mat_struct, contexts, demos_struct = verify_against_matlab(
        data_path, matlab_reference, args.tol
    )
    for name, diff, tol, status in report:
        tag = "OK" if status else "FAIL"
        print(f"[{tag}] {name:<20} max diff {diff:.3e} (tol {tol:.1e})")
    nb_samples = int(mat_struct["nbSamples"])
    nb_repros = int(mat_struct["nbRepros"])
    plot_path = (REPO_ROOT / args.plot_path).resolve()
    render_plot(
        demos_struct,
        contexts,
        py_result["reproductions"],
        nb_samples,
        nb_repros,
        plot_path,
    )
    print(f"Saved TP-GMR plot to {plot_path}")
    if not ok:
        raise SystemExit("Standalone TP-GMR deviates from MATLAB beyond tolerance.")


if __name__ == "__main__":
    main()
