"""Learning utilities for the standalone TP-GMR module."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .model import Demonstration, TPGMRModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = np.finfo(float).tiny


def build_observation_tensor(
    demos: Sequence[Demonstration],
    nb_frames: Optional[int] = None,
    time_scaling: float = 1e-1,
) -> FloatArray:
    """Replicates the MATLAB tensor generation (Data variable)."""

    if not demos:
        raise ValueError("At least one demonstration is required.")
    if nb_frames is None:
        nb_frames = len(demos[0].frames)
    nb_var = demos[0].data.shape[0]
    nb_data = demos[0].nb_data
    tensor = np.zeros((nb_var, nb_frames, nb_data * len(demos)))
    for n, demo in enumerate(demos):
        data = demo.data.copy()
        data[0, :] *= time_scaling
        for m in range(nb_frames):
            frame = demo.frames[m]
            transformed = np.linalg.solve(frame.A, data - frame.b[:, None])
            tensor[:, m, n * nb_data : (n + 1) * nb_data] = transformed
    return tensor


def init_tensor_gmm_time_based(
    data: FloatArray, nb_states: int, diag_reg_factor: float = 1e-4
) -> TPGMRModel:
    """Time-based initialization identical to init_tensorGMM_timeBased.m."""

    nb_var, nb_frames, nb_data_total = data.shape
    data_all = np.reshape(data, (nb_var * nb_frames, nb_data_total), order="F")
    timing = data_all[0, :]
    timing_sep = np.linspace(np.min(timing), np.max(timing), nb_states + 1)

    priors = np.zeros(nb_states)
    mu_blocks = np.zeros((nb_var * nb_frames, nb_states))
    sigma_blocks = np.zeros((nb_var * nb_frames, nb_var * nb_frames, nb_states))

    for i in range(nb_states):
        mask = (timing >= timing_sep[i]) & (timing < timing_sep[i + 1])
        segment = data_all[:, mask]
        if segment.size == 0:
            raise ValueError("Empty time bin during initialization.")
        mu_blocks[:, i] = segment.mean(axis=1)
        centered = segment - mu_blocks[:, i][:, None]
        cov = centered @ centered.T / max(segment.shape[1] - 1, 1)
        cov += np.eye(centered.shape[0]) * diag_reg_factor
        sigma_blocks[:, :, i] = cov
        priors[i] = segment.shape[1]

    priors /= np.sum(priors)

    mu = np.zeros((nb_var, nb_frames, nb_states))
    sigma = np.zeros((nb_var, nb_var, nb_frames, nb_states))
    for m in range(nb_frames):
        rows = slice(m * nb_var, (m + 1) * nb_var)
        mu[:, m, :] = mu_blocks[rows, :]
        sigma[:, :, m, :] = sigma_blocks[rows, rows, :]

    inv_sigma = np.zeros_like(sigma)
    for m in range(nb_frames):
        for i in range(nb_states):
            inv_sigma[:, :, m, i] = np.linalg.inv(sigma[:, :, m, i])

    return TPGMRModel(
        nb_states=nb_states,
        nb_frames=nb_frames,
        nb_var=nb_var,
        Priors=priors,
        Mu=mu,
        Sigma=sigma,
        invSigma=inv_sigma,
        params_diagRegFact=diag_reg_factor,
    )


def _gauss_pdf(
    data: FloatArray, mu: FloatArray, sigma: FloatArray
) -> FloatArray:
    diff = data - mu[:, None]
    solve = np.linalg.solve(sigma, diff)
    quad = np.sum(diff * solve, axis=0)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance matrix not SPD")
    dim = data.shape[0]
    log_norm = 0.5 * (logdet + dim * np.log(2 * np.pi))
    return np.exp(-0.5 * quad - log_norm)


def _compute_gamma(
    data: FloatArray, model: TPGMRModel
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    nb_data = data.shape[2]
    lik = np.ones((model.nb_states, nb_data))
    gamma0 = np.zeros((model.nb_states, model.nb_frames, nb_data))
    for i in range(model.nb_states):
        for m in range(model.nb_frames):
            data_mat = data[:, m, :]
            gamma0[i, m, :] = _gauss_pdf(
                data_mat,
                model.Mu[:, m, i],
                model.Sigma[:, :, m, i],
            )
            lik[i, :] *= gamma0[i, m, :]
        lik[i, :] *= model.Priors[i]
    gamma = lik / (np.sum(lik, axis=0, keepdims=True) + _EPS)
    return lik, gamma, gamma0


def em_tensor_gmm(data: FloatArray, model: TPGMRModel) -> TPGMRModel:
    """Expectation maximization for the task-parameterized tensor GMM."""

    nb_data = data.shape[2]
    prev_ll = None
    for iteration in range(model.params_nbMaxSteps):
        lik, gamma, _ = _compute_gamma(data, model)
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma2 = gamma / (gamma_sum + _EPS)
        model.Pix = gamma2

        if model.params_updateComp[0]:
            model.Priors = np.sum(gamma, axis=1) / nb_data

        for i in range(model.nb_states):
            for m in range(model.nb_frames):
                data_mat = data[:, m, :]
                if model.params_updateComp[1]:
                    model.Mu[:, m, i] = data_mat @ gamma2[i, :]
                if model.params_updateComp[2]:
                    diff = data_mat - model.Mu[:, m, i][:, None]
                    weighted = diff * gamma2[i, :][None, :]
                    cov = weighted @ diff.T + np.eye(data_mat.shape[0]) * model.params_diagRegFact
                    model.Sigma[:, :, m, i] = cov

        for m in range(model.nb_frames):
            for i in range(model.nb_states):
                model.invSigma[:, :, m, i] = np.linalg.inv(model.Sigma[:, :, m, i])

        ll = np.sum(np.log(np.sum(lik, axis=0) + _EPS)) / nb_data
        if iteration + 1 > model.params_nbMinSteps:
            if prev_ll is not None and (ll - prev_ll) < model.params_maxDiffLL:
                break
            if iteration + 1 == model.params_nbMaxSteps - 1:
                break
        prev_ll = ll
    return model


def gmr_time_based(
    model: TPGMRModel,
    data_in: FloatArray,
    out_idx: Sequence[int],
    diag_regularization: Optional[float] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Gaussian mixture regression for each frame given time input."""

    diag_regularization = (
        model.params_diagRegFact if diag_regularization is None else diag_regularization
    )
    nb_data = data_in.shape[1]
    nb_out = len(out_idx)
    mu_gmr = np.zeros((nb_out, nb_data, model.nb_frames))
    sigma_gmr = np.zeros((nb_out, nb_out, nb_data, model.nb_frames))
    in_idx = [0]
    for m in range(model.nb_frames):
        H = np.zeros((model.nb_states, nb_data))
        cache = []
        for i in range(model.nb_states):
            sigma_full = model.Sigma[:, :, m, i]
            mu_full = model.Mu[:, m, i]
            sigma_in_in = sigma_full[np.ix_(in_idx, in_idx)]
            sigma_in_in_inv = np.linalg.inv(sigma_in_in)
            sigma_out_in = sigma_full[np.ix_(out_idx, in_idx)]
            sigma_in_out = sigma_full[np.ix_(in_idx, out_idx)]
            sigma_out_out = sigma_full[np.ix_(out_idx, out_idx)]
            cache.append(
                (
                    mu_full[in_idx],
                    mu_full[out_idx],
                    sigma_in_in_inv,
                    sigma_out_in,
                    sigma_in_out,
                    sigma_out_out,
                )
            )
            H[i, :] = model.Priors[i] * _gauss_pdf(
                data_in,
                mu_full[in_idx],
                sigma_in_in,
            )
        H /= np.sum(H, axis=0, keepdims=True) + _EPS
        for t in range(nb_data):
            mu_tmp = np.zeros((nb_out, model.nb_states))
            for i, (mu_in, mu_out, sigma_in_inv, sigma_out_in, sigma_in_out, sigma_out_out) in enumerate(cache):
                mu_tmp[:, i] = mu_out + sigma_out_in @ (
                    sigma_in_inv @ (data_in[:, t] - mu_in)
                )
                mu_gmr[:, t, m] += H[i, t] * mu_tmp[:, i]
            for i, (_, _, sigma_in_inv, sigma_out_in, sigma_in_out, sigma_out_out) in enumerate(cache):
                sigma_tmp = sigma_out_out - sigma_out_in @ (sigma_in_inv @ sigma_in_out)
                sigma_gmr[:, :, t, m] += H[i, t] * (
                    sigma_tmp + np.outer(mu_tmp[:, i], mu_tmp[:, i])
                )
            sigma_gmr[:, :, t, m] -= np.outer(mu_gmr[:, t, m], mu_gmr[:, t, m])
            sigma_gmr[:, :, t, m] += np.eye(nb_out) * diag_regularization
    return mu_gmr, sigma_gmr

