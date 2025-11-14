#!/usr/bin/env python3
"""Generate synthetic 3D spiral demonstrations for the TP-GMR pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

DEFAULT_OUTPUT = Path("tpgmm_pylib/TPGMR/examples/data/spiral_3d_tpgmr.npz")

FloatArray = NDArray[np.float64]


def rotation_z(angle: float) -> FloatArray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def build_frame(anchor: FloatArray, angle: float, scales: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Create a 4x4 affine matrix (time + 3D position) anchored at a point."""

    rot = rotation_z(angle) @ np.diag(scales)
    A = np.eye(4)
    A[1:, 1:] = rot
    b = np.zeros(4)
    b[1:] = anchor
    return A, b


@dataclass
class SpiralDemo:
    name: str
    duration: float
    time_samples: FloatArray
    positions: FloatArray
    frames_A: FloatArray
    frames_b: FloatArray
    dt: float


def build_demo(
    idx: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> SpiralDemo:
    ctrl_count = args.control_points
    ctrl_t = np.linspace(0.0, 1.0, ctrl_count)
    rotations = args.rotations + 0.12 * idx
    radius = args.radius * (1.0 + 0.12 * idx)
    vertical_span = args.height * (1.0 + 0.08 * idx)
    wobble = rng.normal(scale=0.02, size=(3, ctrl_count))

    theta = 2 * np.pi * rotations * ctrl_t
    ctrl = np.zeros((3, ctrl_count))
    ctrl[0] = radius * np.cos(theta) + wobble[0]
    ctrl[1] = radius * np.sin(theta) + wobble[1]
    ctrl[2] = vertical_span * ctrl_t + 0.1 * np.sin(2 * np.pi * ctrl_t) + wobble[2]

    duration = args.duration + 0.3 * idx
    dense_t = np.linspace(0.0, duration, args.samples)
    spline = CubicSpline(ctrl_t * duration, ctrl, axis=1, bc_type="clamped")
    positions = spline(dense_t)

    dt = duration / max(args.samples - 1, 1)

    name = f"spiral3d_demo_{idx+1:02d}"
    start_anchor = positions[:, 0]
    goal_anchor = positions[:, -1]
    start_A, start_b = build_frame(
        start_anchor,
        angle=-0.05 * idx,
        scales=0.4 + 0.2 * rng.random(3),
    )
    goal_A, goal_b = build_frame(
        goal_anchor,
        angle=0.2 + 0.08 * idx,
        scales=0.4 + 0.2 * rng.random(3),
    )

    time_samples = np.linspace(0.0, duration, args.samples)
    return SpiralDemo(
        name=name,
        duration=duration,
        time_samples=time_samples,
        positions=positions,
        frames_A=np.stack([start_A, goal_A]),
        frames_b=np.stack([start_b, goal_b]),
        dt=dt,
    )


def generate_spiral_dataset(
    out_path: Path,
    nb_demos: int = 3,
    samples: int = 220,
    control_points: int = 7,
    duration: float = 5.0,
    radius: float = 0.22,
    height: float = 0.35,
    rotations: float = 1.0,
    seed: int = 5,
) -> Path:
    """Generate and save a synthetic spiral dataset for TP-GMR."""

    rng = np.random.default_rng(seed)
    args = argparse.Namespace(
        nb_demos=nb_demos,
        samples=samples,
        control_points=control_points,
        duration=duration,
        radius=radius,
        height=height,
        rotations=rotations,
    )

    demos: List[SpiralDemo] = []
    for idx in range(nb_demos):
        demos.append(build_demo(idx, rng, args))

    time_samples = np.stack([demo.time_samples for demo in demos])
    trajectories = np.stack([demo.positions for demo in demos])
    frames_A = np.stack([demo.frames_A for demo in demos])
    frames_b = np.stack([demo.frames_b for demo in demos])
    names = np.array([demo.name for demo in demos])
    durations = np.array([demo.duration for demo in demos])
    dts = np.array([demo.dt for demo in demos])

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        time_samples=time_samples,
        trajectories=trajectories,
        frames_A=frames_A,
        frames_b=frames_b,
        names=names,
        durations=durations,
        dt=float(np.mean(dts)),
        info="Synthetic 3D spiral dataset for TP-GMR",
    )
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Destination NPZ path.")
    parser.add_argument("--nb-demos", type=int, default=3, help="Number of demonstrations.")
    parser.add_argument("--samples", type=int, default=220, help="Samples per demonstration.")
    parser.add_argument("--control-points", type=int, default=7, help="Spline control points per axis.")
    parser.add_argument("--duration", type=float, default=5.0, help="Base trajectory duration.")
    parser.add_argument("--radius", type=float, default=0.22, help="Base radial distance.")
    parser.add_argument("--height", type=float, default=0.35, help="Base vertical span.")
    parser.add_argument("--rotations", type=float, default=1.0, help="Base number of spiral turns.")
    parser.add_argument("--seed", type=int, default=5, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    path = generate_spiral_dataset(
        args.out,
        nb_demos=args.nb_demos,
        samples=args.samples,
        control_points=args.control_points,
        duration=args.duration,
        radius=args.radius,
        height=args.height,
        rotations=args.rotations,
        seed=args.seed,
    )
    print(f"Saved {args.nb_demos} spiral demonstrations to {path}")


if __name__ == "__main__":
    main()
