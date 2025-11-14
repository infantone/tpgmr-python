"""Standalone TP-GMR python port mirroring demo_TPGMR01.m."""
from .datasets import demos_from_arrays, demos_from_npz, load_matlab_demos
from .learning import (
    build_observation_tensor,
    em_tensor_gmm,
    gmr_time_based,
    init_tensor_gmm_time_based,
)
from .model import (
    Demonstration,
    Frame,
    Reproduction,
    TPGMRModel,
    model_to_numpy_dict,
)
from .pipeline import run_tpgmr_demo, run_tpgmr_from_demos
from .reproduction import collect_contexts, reproduce_trajectories

__all__ = [
    "Frame",
    "Demonstration",
    "Reproduction",
    "TPGMRModel",
    "model_to_numpy_dict",
    "load_matlab_demos",
    "demos_from_arrays",
    "demos_from_npz",
    "build_observation_tensor",
    "init_tensor_gmm_time_based",
    "em_tensor_gmm",
    "gmr_time_based",
    "collect_contexts",
    "reproduce_trajectories",
    "run_tpgmr_demo",
    "run_tpgmr_from_demos",
]

