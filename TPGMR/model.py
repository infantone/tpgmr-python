"""Core data structures for the standalone TP-GMR port."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass
class Frame:
    """Affine frame defining a task parameterization."""

    A: FloatArray
    b: FloatArray

    def copy(self) -> "Frame":
        return Frame(A=self.A.copy(), b=self.b.copy())


@dataclass
class Demonstration:
    """One demonstration trajectory along with its frames."""

    data: FloatArray
    frames: List[Frame]

    @property
    def nb_data(self) -> int:
        return int(self.data.shape[1])


@dataclass
class Reproduction:
    """Reproduced trajectory fused across frames."""

    frames: List[Frame]
    data: FloatArray
    sigma: FloatArray
    mu_components: Optional[FloatArray] = None
    sigma_components: Optional[FloatArray] = None


@dataclass
class TPGMRModel:
    """Task-parameterized GMM parameters."""

    nb_states: int
    nb_frames: int
    nb_var: int
    Priors: FloatArray
    Mu: FloatArray
    Sigma: FloatArray
    invSigma: FloatArray
    params_diagRegFact: float = 1e-4
    params_nbMinSteps: int = 5
    params_nbMaxSteps: int = 100
    params_maxDiffLL: float = 1e-5
    params_updateComp: BoolArray = field(
        default_factory=lambda: np.ones(3, dtype=bool)
    )
    Pix: Optional[FloatArray] = None

    def copy(self) -> "TPGMRModel":
        return TPGMRModel(
            nb_states=self.nb_states,
            nb_frames=self.nb_frames,
            nb_var=self.nb_var,
            Priors=self.Priors.copy(),
            Mu=self.Mu.copy(),
            Sigma=self.Sigma.copy(),
            invSigma=self.invSigma.copy(),
            params_diagRegFact=self.params_diagRegFact,
            params_nbMinSteps=self.params_nbMinSteps,
            params_nbMaxSteps=self.params_nbMaxSteps,
            params_maxDiffLL=self.params_maxDiffLL,
            params_updateComp=self.params_updateComp.copy(),
            Pix=None if self.Pix is None else self.Pix.copy(),
        )


def model_to_numpy_dict(model: TPGMRModel) -> dict:
    """Handy helper for serialization/tests."""

    return {
        "nb_states": model.nb_states,
        "nb_frames": model.nb_frames,
        "nb_var": model.nb_var,
        "Priors": model.Priors.copy(),
        "Mu": model.Mu.copy(),
        "Sigma": model.Sigma.copy(),
        "invSigma": model.invSigma.copy(),
    }
