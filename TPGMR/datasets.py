"""Dataset helpers for TP-GMR demos."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .model import Demonstration, Frame

FloatArray = NDArray[np.float64]


def load_matlab_demos(mat_path: str | Path) -> Tuple[List[Demonstration], int]:
    """Load PbDlib demo structures stored in Data02.mat style files."""

    mat = loadmat(Path(mat_path), squeeze_me=True, struct_as_record=False)
    demos_raw = np.atleast_1d(mat["s"])
    demos: List[Demonstration] = []
    for demo in demos_raw:
        frames = []
        for frame in np.atleast_1d(demo.p):
            frames.append(
                Frame(
                    A=np.array(frame.A, dtype=float),
                    b=np.array(frame.b, dtype=float).reshape(-1),
                )
            )
        data = np.array(demo.Data0, dtype=float)
        demos.append(Demonstration(data=data, frames=frames))
    nb_samples = int(mat["nbSamples"])
    return demos, nb_samples


def demos_from_arrays(
    time_samples: FloatArray,
    trajectories: FloatArray,
    frames_A: FloatArray,
    frames_b: FloatArray,
) -> List[Demonstration]:
    """Create demonstrations from numpy arrays (useful for alternative datasets)."""

    nb_demos, nb_points = time_samples.shape
    demos: List[Demonstration] = []
    for idx in range(nb_demos):
        data = np.vstack([time_samples[idx], trajectories[idx]])
        frames: List[Frame] = []
        for m in range(frames_A.shape[1]):
            frames.append(
                Frame(
                    A=frames_A[idx, m].copy(),
                    b=frames_b[idx, m].copy(),
                )
            )
        demos.append(Demonstration(data=data, frames=frames))
    return demos


def demos_from_npz(npz_path: str | Path) -> Tuple[List[Demonstration], Dict[str, Any]]:
    """Load demonstrations that were exported to npz (time, trajectories, frames)."""

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        time_samples = np.array(data["time_samples"], dtype=float)
        trajectories = np.array(data["trajectories"], dtype=float)
        frames_A = np.array(data["frames_A"], dtype=float)
        frames_b = np.array(data["frames_b"], dtype=float)
        metadata = {
            "names": data.get("names"),
            "durations": data.get("durations"),
            "control_points": data.get("control_points"),
            "info": data.get("info"),
        }
    demos = demos_from_arrays(time_samples, trajectories, frames_A, frames_b)
    return demos, metadata

