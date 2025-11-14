"""High level orchestration of the TP-GMR workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .datasets import load_matlab_demos
from .learning import (
    build_observation_tensor,
    em_tensor_gmm,
    gmr_time_based,
    init_tensor_gmm_time_based,
)
from .model import Demonstration, Frame
from .reproduction import collect_contexts, reproduce_trajectories


def run_tpgmr_from_demos(
    demos: Sequence[Demonstration],
    nb_states: int = 3,
    nb_frames: Optional[int] = None,
    nb_repros: int = 4,
    diag_reg_factor: float = 1e-4,
    random_state: Optional[int] = 0,
    contexts_override: Optional[Sequence[Sequence[Frame]]] = None,
    time_scaling: float = 1e-1,
) -> dict:
    """Train TP-GMM + run TP-GMR reproductions based purely on provided demos."""

    if not demos:
        raise ValueError("At least one demonstration is required.")
    if nb_frames is None:
        nb_frames = len(demos[0].frames)

    data = build_observation_tensor(demos, nb_frames=nb_frames, time_scaling=time_scaling)
    model = init_tensor_gmm_time_based(data, nb_states, diag_reg_factor)
    model = em_tensor_gmm(data, model)
    data_in = demos[0].data[[0], :].copy() * time_scaling
    out_idx = list(range(1, model.nb_var))
    mu_gmr, sigma_gmr = gmr_time_based(model, data_in, out_idx)

    if contexts_override is not None:
        contexts = [[frame.copy() for frame in frames] for frames in contexts_override]
    else:
        contexts = collect_contexts(
            demos,
            nb_repros=nb_repros,
            random_state=random_state,
        )
    reproductions = reproduce_trajectories(model, mu_gmr, sigma_gmr, contexts)
    return {
        "Data": data,
        "DataIn": data_in,
        "model": model,
        "MuGMR": mu_gmr,
        "SigmaGMR": sigma_gmr,
        "reproductions": reproductions,
        "nbData": demos[0].nb_data,
        "nbSamples": len(demos),
        "nbRepros": nb_repros,
    }


def run_tpgmr_demo(
    data_path: str | Path,
    nb_states: int = 3,
    nb_frames: int = 2,
    nb_repros: int = 4,
    diag_reg_factor: float = 1e-4,
    random_state: Optional[int] = 0,
    contexts_override: Optional[Sequence[Sequence[Frame]]] = None,
    time_scaling: float = 1e-1,
) -> dict:
    """Convenience wrapper that mirrors demo_TPGMR01.m exactly."""

    demos, nb_samples = load_matlab_demos(data_path)
    result = run_tpgmr_from_demos(
        demos,
        nb_states=nb_states,
        nb_frames=nb_frames,
        nb_repros=nb_repros,
        diag_reg_factor=diag_reg_factor,
        random_state=random_state,
        contexts_override=contexts_override,
        time_scaling=time_scaling,
    )
    result["nbSamples"] = nb_samples
    return result
