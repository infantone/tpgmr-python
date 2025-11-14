"""Context sampling and reproduction helpers."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .model import Demonstration, Frame, Reproduction, TPGMRModel

FloatArray = NDArray[np.float64]


def collect_contexts(
    demos: Sequence[Demonstration],
    nb_repros: int,
    random_state: Optional[int] = None,
) -> List[List[Frame]]:
    """Collect demo frames and optionally sample extra random contexts."""

    rng = np.random.default_rng(random_state)
    contexts: List[List[Frame]] = [
        [frame.copy() for frame in demo.frames] for demo in demos
    ]
    nb_samples = len(demos)
    nb_frames = len(demos[0].frames)
    for _ in range(nb_repros):
        frames: List[Frame] = []
        for m in range(nb_frames):
            idx = rng.integers(0, nb_samples, size=2)
            weights = rng.random(2)
            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                weights = np.array([0.5, 0.5])
                weights_sum = 1.0
            weights /= weights_sum
            frame_a = demos[idx[0]].frames[m]
            frame_b = demos[idx[1]].frames[m]
            frames.append(
                Frame(
                    A=frame_a.A * weights[0] + frame_b.A * weights[1],
                    b=frame_a.b * weights[0] + frame_b.b * weights[1],
                )
            )
        contexts.append(frames)
    return contexts


def reproduce_trajectories(
    model: TPGMRModel,
    mu_gmr: FloatArray,
    sigma_gmr: FloatArray,
    contexts: Sequence[Sequence[Frame]],
) -> List[Reproduction]:
    """Apply frame transformations and fuse Gaussians across frames."""

    nb_out, nb_data, nb_frames = mu_gmr.shape
    if nb_frames != model.nb_frames:
        raise ValueError("Mismatch between model frames and MuGMR tensor.")
    reproductions: List[Reproduction] = []
    for frames in contexts:
        mu_tmp = np.zeros((nb_out, nb_data, nb_frames))
        sigma_tmp = np.zeros((nb_out, nb_out, nb_data, nb_frames))
        for m, frame in enumerate(frames):
            A = frame.A[1:, 1:]
            b = frame.b[1:]
            mu_tmp[:, :, m] = A @ mu_gmr[:, :, m] + b[:, None]
            for t in range(nb_data):
                sigma_tmp[:, :, t, m] = A @ sigma_gmr[:, :, t, m] @ A.T
        traj = np.zeros((nb_out, nb_data))
        covs = np.zeros((nb_out, nb_out, nb_data))
        for t in range(nb_data):
            sigma_inv = np.zeros((nb_out, nb_out))
            mu_acc = np.zeros(nb_out)
            for m in range(nb_frames):
                inv_tmp = np.linalg.inv(sigma_tmp[:, :, t, m])
                sigma_inv += inv_tmp
                mu_acc += inv_tmp @ mu_tmp[:, t, m]
            covs[:, :, t] = np.linalg.inv(sigma_inv)
            traj[:, t] = covs[:, :, t] @ mu_acc
        mu_components, sigma_components = fuse_context_gaussians(model, frames)
        reproductions.append(
            Reproduction(
                frames=[frame.copy() for frame in frames],
                data=traj,
                sigma=covs,
                mu_components=mu_components,
                sigma_components=sigma_components,
            )
        )
    return reproductions


def fuse_context_gaussians(
    model: TPGMRModel,
    frames: Sequence[Frame],
    weights: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Fuse model Gaussians for a specific context."""

    nb_var = model.nb_var
    nb_states = model.nb_states
    mu = np.zeros((nb_var, nb_states))
    sigma = np.zeros((nb_var, nb_var, nb_states))
    for i in range(nb_states):
        sigma_tmp = np.zeros((nb_var, nb_var))
        mu_tmp = np.zeros(nb_var)
        for m, frame in enumerate(frames):
            w = 1.0
            if weights is not None:
                w = float(weights[i, m])
                if w <= 0:
                    continue
            mu_frame = frame.A @ model.Mu[:, m, i] + frame.b
            sigma_frame = frame.A @ model.Sigma[:, :, m, i] @ frame.A.T
            sigma_inv = np.linalg.inv(sigma_frame)
            sigma_tmp += w * sigma_inv
            mu_tmp += w * (sigma_inv @ mu_frame)
        if not np.any(sigma_tmp):
            raise ValueError("Degenerate fusion: frame weights produced singular covariance.")
        sigma[:, :, i] = np.linalg.inv(sigma_tmp)
        mu[:, i] = sigma[:, :, i] @ mu_tmp
    return mu, sigma
