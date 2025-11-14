#!/usr/bin/env python3
"""Interactive TP-GMR training/generalization for recorded robot trajectories."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import ast
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TPGMR import run_tpgmr_from_demos
from TPGMR.model import Demonstration, Frame
from TPGMR.learning import (
    build_observation_tensor,
    em_tensor_gmm,
    init_tensor_gmm_time_based,
)


DEFAULT_DEMO_DIR = Path("TPGMR/robot_trajectories/demo")
DEFAULT_OUTPUT_DIR = Path("TPGMR/robot_trajectories/generalized")
DEFAULT_CONFIG_DIR = Path("TPGMR/robot_trajectories/config")

_PYLAB = None
_MATPLOTLIB = None
_ANIMATIONS: list = []


def _get_pyplot(interactive: bool):
    """Ottiene pyplot con il backend corretto."""
    global _PYLAB, _MATPLOTLIB
    if _PYLAB is not None:
        return _PYLAB
    
    # Import matplotlib solo quando necessario
    import matplotlib
    _MATPLOTLIB = matplotlib
    
    if interactive:
        # Forza un backend interattivo PRIMA di importare pyplot
        backend_set = False
        for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "Qt4Agg", "WXAgg", "MacOSX"):
            try:
                matplotlib.use(candidate, force=True)
                backend_set = True
                print(f"Usando backend matplotlib: {candidate}")
                break
            except Exception:
                continue
        
        if not backend_set:
            print("ATTENZIONE: Nessun backend interattivo disponibile, provo con il default")
            try:
                matplotlib.use("TkAgg", force=True)
            except:
                pass
    else:
        matplotlib.use("Agg")
    
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import animation  # noqa: F401
    from matplotlib.lines import Line2D  # noqa: F401
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    
    # Salva i moduli nel namespace globale per usarli nelle altre funzioni
    globals()['animation'] = animation
    globals()['Line2D'] = Line2D
    globals()['Poly3DCollection'] = Poly3DCollection

    _PYLAB = plt
    return plt


def _ensure_monotonic_time(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time_sorted = time[order]
    values_sorted = values[:, order]
    diffs = np.diff(time_sorted)
    mask = np.ones_like(time_sorted, dtype=bool)
    mask[1:] = diffs > 1e-9
    time_filtered = time_sorted[mask]
    time_filtered = time_filtered - time_filtered[0]
    return time_filtered, values_sorted[:, mask]


def _build_time_grid(time: np.ndarray, samples: int | None) -> np.ndarray:
    if time.shape[0] < 2:
        raise ValueError("Una demo deve contenere almeno due campioni temporali.")
    if samples is None or samples <= 0:
        return time.copy()
    duration = float(time[-1])
    if duration <= 0:
        raise ValueError("Demo duration must be positive.")
    return np.linspace(0.0, duration, samples)


def _resample_channels(time_src: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if values.shape[1] != time_src.shape[0]:
        raise ValueError("Dimensione dei dati incoerente con il vettore tempo sorgente.")
    if time_src.shape[0] == target_time.shape[0] and np.allclose(time_src, target_time):
        return values.copy()
    result = np.zeros((values.shape[0], target_time.shape[0]))
    for idx in range(values.shape[0]):
        spline = CubicSpline(time_src, values[idx])
        result[idx] = spline(target_time)
    return result


def _resample_channels_phase(time_src: np.ndarray, values: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    duration = float(time_src[-1]) if time_src.size else 0.0
    if duration <= 0:
        return np.repeat(values[:, :1], target_phase.shape[0], axis=1)
    phase_src = time_src / duration
    result = np.zeros((values.shape[0], target_phase.shape[0]))
    for idx in range(values.shape[0]):
        spline = CubicSpline(phase_src, values[idx])
        result[idx] = spline(target_phase)
    return result


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Zero-norm quaternion encountered.")
    return q / norm


def _enforce_quat_continuity(quats: np.ndarray) -> np.ndarray:
    adjusted = quats.copy()
    for idx in range(1, adjusted.shape[0]):
        if np.dot(adjusted[idx - 1], adjusted[idx]) < 0.0:
            adjusted[idx] *= -1.0
    return adjusted


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return _quat_normalize(quat)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def _slerp(q0: np.ndarray, q1: np.ndarray, weights: np.ndarray) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        lerp = q0[None, :] + weights[:, None] * (q1 - q0)[None, :]
        lerp /= np.linalg.norm(lerp, axis=1, keepdims=True)
        return lerp
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    sin_theta = np.sin(theta_0 * weights)
    sin_theta_1 = np.sin(theta_0 * (1.0 - weights))
    return (
        (sin_theta_1[:, None] * q0[None, :] + sin_theta[:, None] * q1[None, :]) / sin_theta_0
    )


def _smooth_derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if values.shape[-1] != time.shape[0]:
        raise ValueError("Dimensione dei campioni incoerente con il vettore tempo.")
    if time.shape[0] < 2:
        raise ValueError("Servono almeno due campioni temporali per calcolare la derivata.")
    result = np.zeros_like(values)
    for idx in range(values.shape[0]):
        spline = CubicSpline(time, values[idx])
        result[idx] = spline(time, 1)
    return result


def _angular_velocity(quats: np.ndarray, time: np.ndarray) -> np.ndarray:
    quats = _enforce_quat_continuity(quats)
    dq_dt = _smooth_derivative(quats.T, time).T
    omega = np.zeros((quats.shape[0], 3))
    for idx, (quat, dq) in enumerate(zip(quats, dq_dt)):
        prod = _quat_multiply(2.0 * dq, _quat_conjugate(quat))
        omega[idx] = prod[1:]
    return omega


def _arrow_from_frame(ax, frame: Frame, length: float = 0.12) -> None:
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


def _ellipsoid_mesh(center: np.ndarray, cov: np.ndarray, n_phi: int = 24, n_theta: int = 12):
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
    ellip = transform @ sphere + center[:, None]
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


def _plot_component_ellipsoids(ax, reproduction, color: np.ndarray, alpha: float = 0.12):
    mu = reproduction.mu_components
    sigma = reproduction.sigma_components
    if mu is None or sigma is None:
        return
    mu = mu[1:, :]
    sigma = sigma[1:, 1:, :]
    for idx in range(mu.shape[1]):
        faces = _ellipsoid_mesh(mu[:, idx], sigma[:, :, idx])
        collection = Poly3DCollection(faces, facecolor=color, alpha=alpha, linewidths=0.0)
        ax.add_collection3d(collection)


def _scatter_keypoints(ax, traj: np.ndarray, color: np.ndarray):
    ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0], marker="*", color=color, s=40)
    ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], marker="o", color=color, s=35)


@dataclass
class RobotDemo:
    path: Path
    name: str
    demo: Demonstration
    time: np.ndarray
    phase: np.ndarray
    frames: List[Frame]
    wrench: np.ndarray
    gripper: np.ndarray


def _load_robot_demo(
    path: Path,
    target_time: np.ndarray | None,
    target_phase: np.ndarray | None,
    requested_samples: int | None = None,
) -> RobotDemo:
    data = np.load(path, allow_pickle=False)
    pose = np.array(data["pose"], dtype=float)
    time = np.array(data["time"], dtype=float)
    time_sorted, pose_sorted = _ensure_monotonic_time(time, pose)
    positions_sorted = pose_sorted[:3]
    duration = float(time_sorted[-1]) if time_sorted.size else 0.0
    if target_time is None:
        time_grid = _build_time_grid(time_sorted, requested_samples)
        phase_grid = time_grid / duration if duration > 0 else np.linspace(0.0, 1.0, time_grid.shape[0])
        resampled_pos = _resample_channels(time_sorted, positions_sorted, time_grid)
    else:
        time_grid = target_time.copy()
        if target_phase is None:
            raise ValueError("target_phase richiesto quando si fornisce target_time.")
        phase_grid = target_phase.copy()
        resampled_pos = _resample_channels_phase(time_sorted, positions_sorted, phase_grid)
        resample_fn = _resample_channels_phase
        resample_args = (time_sorted,)
    start_quat_xyzw = pose_sorted[3:, 0]
    goal_quat_xyzw = pose_sorted[3:, -1]
    start_frame = _build_frame(resampled_pos[:, 0], start_quat_xyzw)
    goal_frame = _build_frame(resampled_pos[:, -1], goal_quat_xyzw)
    demo_data = np.vstack([time_grid, resampled_pos])
    wrench = np.array(data["wrench"], dtype=float)
    wrench_resampled = (
        _resample_channels(time_sorted, wrench, time_grid)
        if target_time is None
        else _resample_channels_phase(time_sorted, wrench, phase_grid)
    )
    gripper = np.array(data["gripper"], dtype=float).reshape(1, -1)
    gripper_resampled = (
        _resample_channels(time_sorted, gripper, time_grid)[0]
        if target_time is None
        else _resample_channels_phase(time_sorted, gripper, phase_grid)[0]
    )
    demo = Demonstration(data=demo_data, frames=[start_frame, goal_frame])
    return RobotDemo(
        path=path,
        name=path.stem,
        demo=demo,
        time=time_grid,
        phase=phase_grid,
        frames=[start_frame, goal_frame],
        wrench=wrench_resampled,
        gripper=gripper_resampled,
    )


def _build_frame(position: np.ndarray, quat_xyzw: np.ndarray) -> Frame:
    frame_A = np.eye(4)
    frame_b = np.zeros(4)
    frame_b[1:] = position
    frame_A[1:, 1:] = _quat_to_matrix(_quat_xyzw_to_wxyz(quat_xyzw))
    return Frame(A=frame_A, b=frame_b)


def _load_robot_demo_group(paths: Sequence[Path], requested_samples: int | None) -> List[RobotDemo]:
    if not paths:
        raise ValueError("Nessuna demo selezionata.")
    demos: List[RobotDemo] = []
    reference_demo = _load_robot_demo(
        paths[0],
        target_time=None,
        target_phase=None,
        requested_samples=requested_samples,
    )
    demos.append(reference_demo)
    for path in paths[1:]:
        demos.append(
            _load_robot_demo(
                path,
                target_time=reference_demo.time,
                target_phase=reference_demo.phase,
            )
        )
    return demos


def _frame_from_transform(transform: Sequence[float]) -> Frame:
    matrix = np.array(transform, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    translation = matrix[3, :3]
    quat_wxyz = _matrix_to_quat(rotation)
    quat_xyzw = _quat_wxyz_to_xyzw(quat_wxyz)
    return _build_frame(translation, quat_xyzw)


def _load_yaml_transform(path: Path) -> np.ndarray:
    text = path.read_text()
    data = None
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = None
    if not data or "O_T_EE" not in data:
        match = re.search(r"O_T_EE\s*:\s*(\[[^\]]+\])", text, re.DOTALL)
        if not match:
            raise ValueError(f"Unable to find O_T_EE in {path}")
        data = {"O_T_EE": ast.literal_eval(match.group(1))}
    transform = np.array(data["O_T_EE"], dtype=float)
    if transform.size != 16:
        raise ValueError(f"O_T_EE in {path} must contain 16 values.")
    return transform


def _list_config_frames(config_dir: Path) -> List[tuple[str, Frame, Path]]:
    frames: List[tuple[str, Frame, Path]] = []
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            transform = _load_yaml_transform(yaml_path)
            frame = _frame_from_transform(transform)
            frames.append((yaml_path.stem, frame, yaml_path))
        except Exception as exc:
            print(f"Impossibile caricare {yaml_path.name}: {exc}", file=sys.stderr)
    return frames


def _fetch_realtime_robot_frame(timeout: float = 5.0) -> Frame:
    try:
        import rospy  # type: ignore
        from franka_msgs.msg import FrankaState  # type: ignore
    except Exception as exc:  # pragma: no cover - hardware dependency
        raise RuntimeError("ROS environment with franka_msgs is required for live robot frames.") from exc

    if not rospy.core.is_initialized():  # pragma: no cover - requires ROS master
        rospy.init_node("tpgmr_robot_frame_fetch", anonymous=True, disable_signals=True)
    topic = "/franka_state_controller/franka_states"
    state = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
    return _frame_from_transform(state.O_T_EE)


def _list_demo_files(demo_dir: Path) -> List[Path]:
    return sorted(demo_dir.glob("*.npz"))


def _gauss_pdf_local(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    diff = data - mu[:, None]
    solve = np.linalg.solve(sigma, diff)
    quad = np.sum(diff * solve, axis=0)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance matrix not SPD.")
    dim = data.shape[0]
    log_norm = 0.5 * (logdet + dim * np.log(2 * np.pi))
    return np.exp(-0.5 * quad - log_norm)


def _log_likelihood_for_model(data: np.ndarray, model) -> float:
    nb_data = data.shape[2]
    tiny = np.finfo(float).tiny
    lik = np.ones((model.nb_states, nb_data))
    for i in range(model.nb_states):
        for m in range(model.nb_frames):
            data_mat = data[:, m, :]
            lik[i, :] *= _gauss_pdf_local(
                data_mat,
                model.Mu[:, m, i],
                model.Sigma[:, :, m, i],
            )
        lik[i, :] *= model.Priors[i]
    return float(np.sum(np.log(np.sum(lik, axis=0) + tiny)))


def _auto_select_nb_states(
    demos: Sequence[RobotDemo],
    diag_reg: float,
    time_scaling: float,
) -> int:
    if not demos:
        raise ValueError("At least one demo is required for auto state selection.")
    tensor = build_observation_tensor(
        [demo.demo for demo in demos],
        nb_frames=len(demos[0].frames),
        time_scaling=time_scaling,
    )
    nb_data = tensor.shape[2]
    if nb_data < 5:
        return 2
    max_candidate = max(2, min(12, nb_data // 8))
    candidates = list(range(2, max_candidate + 1))
    best_bic = None
    best_state = candidates[0]
    nb_var = demos[0].demo.data.shape[0]
    nb_frames = len(demos[0].frames)
    for k in candidates:
        model = init_tensor_gmm_time_based(tensor, k, diag_reg)
        model = em_tensor_gmm(tensor, model)
        ll = _log_likelihood_for_model(tensor, model)
        per_frame_params = nb_var + (nb_var * (nb_var + 1) / 2.0)
        nb_params = (k - 1) + k * nb_frames * per_frame_params
        bic = -2.0 * ll + nb_params * np.log(nb_data)
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_state = k
    print(f"Auto-selected {best_state} GMM states using BIC over candidates {candidates}.")
    return best_state


def _parse_selection(selection: str, files: Sequence[Path]) -> List[int]:
    if not selection or selection.lower() == "all":
        return list(range(len(files)))
    parts = [p.strip() for p in selection.split(",") if p.strip()]
    indices = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"Indice non valido: {part}")
        idx = int(part)
        if idx < 1 or idx > len(files):
            raise ValueError(f"Indice fuori range: {idx}")
        indices.append(idx - 1)
    if not indices:
        raise ValueError("Nessuna demo selezionata.")
    return sorted(set(indices))


def _prompt_indices(files: Sequence[Path]) -> List[int]:
    print("Traiettorie disponibili:")
    for idx, path in enumerate(files, start=1):
        print(f"  {idx}) {path.name}")
    selection = input("Seleziona le demo (es. 1,3,4 oppure 'all'): ").strip()
    return _parse_selection(selection, files)


def _choose_context(name: str, options: Sequence[RobotDemo], default_idx: int, user_choice: int | None) -> int:
    if user_choice is not None:
        idx = user_choice - 1
        if idx < 0 or idx >= len(options):
            raise ValueError(f"Indice {user_choice} non valido per il frame {name}.")
        return idx
    prompt = f"Seleziona la demo per il frame {name} [{default_idx + 1}]: "
    choice = input(prompt).strip()
    if not choice:
        return default_idx
    if not choice.isdigit():
        raise ValueError(f"Indice non valido: {choice}")
    idx = int(choice) - 1
    if idx < 0 or idx >= len(options):
        raise ValueError("Indice fuori range.")
    return idx


def _build_output_filename(output_dir: Path, custom_name: str | None = None) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir.expanduser()
    if custom_name:
        name_clean = custom_name.strip()
        if name_clean:
            base = Path(name_clean).name
            if not base.lower().endswith(".npz"):
                base = f"{base}.npz"
            return output_dir / base
    return output_dir / f"generalized_{timestamp}.npz"


def _save_generalization(
    output_path: Path,
    reproduction,
    time_scaled: np.ndarray,
    time_scaling: float,
    start_frame: Frame,
    goal_frame: Frame,
    reference_time: np.ndarray,
    reference_wrench: np.ndarray,
    reference_gripper: np.ndarray,
) -> None:
    positions = reproduction.data
    nb_points = positions.shape[1]
    time = time_scaled / time_scaling
    start_quat = _matrix_to_quat(start_frame.A[1:, 1:])
    goal_quat = _matrix_to_quat(goal_frame.A[1:, 1:])
    weights = np.linspace(0.0, 1.0, nb_points)
    quats_wxyz = _slerp(start_quat, goal_quat, weights)
    quats_wxyz = _enforce_quat_continuity(quats_wxyz)
    quats_xyzw = np.vstack([_quat_wxyz_to_xyzw(q) for q in quats_wxyz]).T
    pose = np.vstack([positions, quats_xyzw])

    linear_vel = _smooth_derivative(positions, time)
    angular_vel = _angular_velocity(quats_wxyz, time).T
    twist = np.vstack([linear_vel, angular_vel])

    wrench = reference_wrench.copy()
    if wrench.shape[1] != nb_points or not np.allclose(reference_time, time):
        wrench = _resample_channels(reference_time, reference_wrench, time)
    joint_pos = np.zeros((7, nb_points))
    joint_vel = np.zeros((7, nb_points))
    gripper = reference_gripper.copy()
    if gripper.shape[0] != nb_points or not np.allclose(reference_time, time):
        gripper = _resample_channels(reference_time, reference_gripper.reshape(1, -1), time)[0]
    else:
        gripper = gripper.astype(int)
    goal_pose = pose[:, -1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        pose=pose,
        twist=twist,
        wrench=wrench,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        gripper=gripper,
        time=time,
        goal=goal_pose,
        start_frame_A=start_frame.A,
        start_frame_b=start_frame.b,
        goal_frame_A=goal_frame.A,
        goal_frame_b=goal_frame.b,
    )
    print(f"Traiettoria generalizzata salvata in {output_path}")


def _load_saved_generalization(path: Path):
    data = np.load(path, allow_pickle=False)
    pose = np.array(data["pose"], dtype=float)
    time = np.array(data["time"], dtype=float)

    def _frame_from_npz(prefix: str) -> Frame:
        if f"{prefix}_A" in data and f"{prefix}_b" in data:
            return Frame(A=np.array(data[f"{prefix}_A"], dtype=float), b=np.array(data[f"{prefix}_b"], dtype=float))
        position = pose[:3, 0] if prefix == "start_frame" else pose[:3, -1]
        quat = pose[3:, 0] if prefix == "start_frame" else pose[3:, -1]
        return _build_frame(position, quat)

    start_frame = _frame_from_npz("start_frame")
    goal_frame = _frame_from_npz("goal_frame")
    reproduction = SimpleNamespace(
        data=pose[:3, :],
        frames=[start_frame, goal_frame],
        mu_components=None,
        sigma_components=None,
    )
    return reproduction, time


def _plot_saved_generalization(path: Path, interactive: bool, animate: bool) -> None:
    reproduction, time = _load_saved_generalization(path)
    _plot_reproduction([], reproduction, interactive, None, animate, time, reproduction.frames)


def _plot_reproduction(
    demos: Sequence[RobotDemo],
    reproduction,
    interactive: bool,
    plot_path: Path | None,
    animate: bool,
    time: np.ndarray | None,
    frames: Sequence[Frame] | None,
) -> None:
    plt = _get_pyplot(interactive)
    if animate and not interactive:
        print("L'animazione richiede anche --interactive.", file=sys.stderr)
        animate = False
    fig = plt.figure(figsize=(13, 6))
    ax_train = fig.add_subplot(121, projection="3d")
    ax_repro = fig.add_subplot(122, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(demos), 3)))

    for idx, demo in enumerate(demos):
        color = colors[idx % len(colors)]
        traj = demo.demo.data[1:4]
        ax_train.plot(traj[0], traj[1], traj[2], "--", color=color, alpha=0.5, label=f"Demo {demo.name}")
        _scatter_keypoints(ax_train, traj, color)
        _arrow_from_frame(ax_train, demo.frames[0], length=0.08)
        _arrow_from_frame(ax_train, demo.frames[-1], length=0.08)

    traj = reproduction.data
    ax_repro.plot(traj[0], traj[1], traj[2], color="#222222", linewidth=2.0)
    _scatter_keypoints(ax_repro, traj, np.array([0.1, 0.1, 0.1]))
    _arrow_from_frame(ax_repro, reproduction.frames[0], length=0.1)
    _arrow_from_frame(ax_repro, reproduction.frames[-1], length=0.1)
    _plot_component_ellipsoids(ax_repro, reproduction, color=np.array([0.3, 0.5, 0.9]), alpha=0.18)

    for ax, title in zip((ax_train, ax_repro), ("Demo selezionate", "Generalizzazione")):
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.grid(True, alpha=0.25)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])

    legend_handles = [
        Line2D([], [], linestyle="--", color="0.4", label="Demo"),
        Line2D([], [], linestyle="-", color="#222222", label="Generalizzazione"),
        Line2D([], [], linestyle="None", marker="*", color="black", label="Start"),
        Line2D([], [], linestyle="None", marker="o", color="black", label="Goal"),
    ]
    ax_train.legend(handles=legend_handles, loc="upper left", frameon=False)

    anim = None
    if animate:
        if not interactive or time is None or frames is None:
            print("L'animazione richiede --interactive e le informazioni di tempo/frame.", file=sys.stderr)
        else:
            # Verifica che abbiamo un backend interattivo
            current_backend = _MATPLOTLIB.get_backend().lower() if _MATPLOTLIB else "unknown"
            print(f"Backend corrente: {current_backend}")
            
            # Backend non interattivo: solo "agg" puro, non "tkagg", "qt5agg", ecc.
            if current_backend == "agg":
                print("Backend non interattivo: impossibile mostrare l'animazione.", file=sys.stderr)
            else:
                # Calcola i quaternioni interpolati per ogni punto della traiettoria
                start_quat = _matrix_to_quat(frames[0].A[1:, 1:])
                goal_quat = _matrix_to_quat(frames[-1].A[1:, 1:])
                nb_points_total = reproduction.data.shape[1]
                
                # Riduci il numero di frame per l'animazione (max 150 frame per fluidità)
                # Questo mantiene la durata reale ma riduce il carico computazionale
                max_anim_frames = 150
                nb_anim_frames = min(max_anim_frames, nb_points_total)
                
                # Crea gli indici dei frame da animare (campionamento uniforme)
                anim_indices = np.linspace(0, nb_points_total - 1, nb_anim_frames, dtype=int)
                
                # Calcola i quaternioni per tutti i punti
                weights_all = np.linspace(0.0, 1.0, nb_points_total)
                quats_all = _enforce_quat_continuity(_slerp(start_quat, goal_quat, weights_all))
                
                # Crea elementi grafici per l'animazione
                progress_line, = ax_repro.plot([], [], [], color="#d62728", linewidth=3.0, label="Progresso")
                current_point, = ax_repro.plot([], [], [], 'o', color="#d62728", markersize=10)
                
                # Lista per contenere le frecce del frame (assi X, Y, Z)
                frame_arrows = []
                
                def init_anim():
                    """Inizializza l'animazione"""
                    progress_line.set_data([], [])
                    progress_line.set_3d_properties([])
                    current_point.set_data([], [])
                    current_point.set_3d_properties([])
                    return [progress_line, current_point]
                
                def update_anim(frame_num: int):
                    """Aggiorna l'animazione al frame frame_num"""
                    # Ottieni l'indice reale del punto nella traiettoria
                    idx = anim_indices[frame_num]
                    
                    # Aggiorna la linea del progresso (traiettoria percorsa fino ad ora)
                    seg = slice(0, idx + 1)
                    progress_line.set_data(reproduction.data[0, seg], reproduction.data[1, seg])
                    progress_line.set_3d_properties(reproduction.data[2, seg])
                    
                    # Aggiorna il punto corrente
                    current_point.set_data([reproduction.data[0, idx]], [reproduction.data[1, idx]])
                    current_point.set_3d_properties([reproduction.data[2, idx]])
                    
                    # Rimuovi le frecce precedenti
                    for arrow in frame_arrows:
                        arrow.remove()
                    frame_arrows.clear()
                    
                    # Disegna il frame corrente (assi X, Y, Z)
                    pos = reproduction.data[:, idx]
                    rot = _quat_to_matrix(quats_all[idx])
                    length = 0.05
                    colors_arr = ["#d62728", "#2ca02c", "#1f77b4"]  # Rosso, Verde, Blu per X, Y, Z
                    
                    for axis_idx in range(3):
                        axis_vec = rot @ (np.eye(3)[:, axis_idx] * length)
                        arrow = ax_repro.quiver(
                            pos[0], pos[1], pos[2],
                            axis_vec[0], axis_vec[1], axis_vec[2],
                            color=colors_arr[axis_idx],
                            linewidth=2.0,
                            arrow_length_ratio=0.3,
                        )
                        frame_arrows.append(arrow)
                    
                    return [progress_line, current_point] + frame_arrows
                
                # Calcola l'intervallo per mantenere la durata temporale reale
                total_duration_sec = time[-1] - time[0]
                # Intervallo tra frame = durata totale / numero di frame
                interval_ms = int((total_duration_sec * 1000) / nb_anim_frames)
                # Limita l'intervallo minimo a 16ms (circa 60 FPS) per una buona fluidità
                interval_ms = max(16, interval_ms)
                
                print(f"Animazione: {nb_anim_frames} frames (da {nb_points_total} punti totali)")
                print(f"Durata reale: {total_duration_sec:.2f}s, intervallo: {interval_ms}ms per frame")
                
                anim = animation.FuncAnimation(
                    fig,
                    update_anim,
                    frames=nb_anim_frames,
                    init_func=init_anim,
                    interval=interval_ms,
                    blit=False,  # Usa False per compatibilità con 3D
                    repeat=True,
                )
                _ANIMATIONS.append(anim)

    fig.suptitle("Generalizzazione TP-GMR su traiettorie robot")
    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=170)
        print(f"Plot salvato in {plot_path}")
    
    # Mostra il plot se siamo in modalità interattiva
    if interactive and _MATPLOTLIB:
        backend = _MATPLOTLIB.get_backend().lower()
        # Verifica che NON sia solo "agg" (senza suffissi come "tkagg", "qt5agg", ecc.)
        if backend != "agg":
            print("Mostrando il plot interattivo... (chiudi la finestra per terminare)")
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR, help="Cartella con le demo .npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Cartella di salvataggio")
    parser.add_argument("--nb-states", type=int, default=8, help="Numero di componenti GMM")
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=0,
        help="Campioni desiderati dopo il resampling (0 = usa la lunghezza della demo di riferimento)",
    )
    parser.add_argument("--diag-reg", type=float, default=1e-4, help="Regolarizzazione delle covarianze")
    parser.add_argument("--time-scaling", type=float, default=1e-1, help="Fattore di scaling temporale")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed per eventuali operazioni stocastiche")
    parser.add_argument(
        "--select",
        type=str,
        help="Selezione non interattiva (es. '1,3' oppure 'all').",
    )
    parser.add_argument(
        "--start-context",
        type=int,
        help="Indice 1-based della demo da usare come frame di partenza.",
    )
    parser.add_argument(
        "--goal-context",
        type=int,
        help="Indice 1-based della demo da usare come frame goal.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mostra un plot 3D interattivo con demo e generalizzazione.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Percorso opzionale per salvare il plot (PNG).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Nome file (senza percorso) per la traiettoria generalizzata.",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Mostra un'animazione 3D della posa lungo la traiettoria.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Avvia un'interfaccia grafica per impostare e lanciare la generalizzazione.",
    )
    return parser.parse_args(argv)


def _launch_gui(defaults: argparse.Namespace) -> None:  # pragma: no cover - GUI code
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise SystemExit("Tkinter is required to launch the GUI (--gui).") from exc

    class RobotGeneralizationGUI:
        def __init__(self, root: "tk.Tk", defaults: argparse.Namespace) -> None:
            self.root = root
            self._tk = tk
            self._ttk = ttk
            self.defaults = defaults
            self.root.title("TP-GMR Robot Generalization")
            self.root.geometry("860x780")
            self.root.minsize(800, 700)
            self.root.columnconfigure(0, weight=1)

            self.demo_paths: List[Path] = []
            self.config_frames = _list_config_frames(DEFAULT_CONFIG_DIR)
            self.start_source_map: dict[str, tuple[str, str | None]] = {}
            self.goal_source_map: dict[str, tuple[str, object]] = {}

            self.demo_dir_var = tk.StringVar(value=str(defaults.demo_dir))
            self.output_dir_var = tk.StringVar(value=str(defaults.output_dir))
            self.plot_out_var = tk.StringVar(value=str(defaults.plot_out) if defaults.plot_out else "")
            self.output_name_var = tk.StringVar()
            self.nb_states_var = tk.StringVar(value=str(defaults.nb_states))
            self.nb_samples_var = tk.StringVar(value=str(defaults.nb_samples))
            self.diag_reg_var = tk.StringVar(value=str(defaults.diag_reg))
            self.time_scaling_var = tk.StringVar(value=str(defaults.time_scaling))
            self.random_seed_var = tk.StringVar(value=str(defaults.random_seed))
            self.interactive_var = tk.BooleanVar(value=bool(defaults.interactive))
            self.animation_var = tk.BooleanVar(value=bool(defaults.animation))
            self.start_var = tk.StringVar()
            self.goal_var = tk.StringVar()
            self.status_var = tk.StringVar(value="Select demonstrations to begin.")

            self._build_layout(ttk, tk, filedialog, messagebox)
            self._on_interactive_toggle()
            self._set_context_controls_state(False)

        def _build_layout(self, ttk, tk, filedialog, messagebox):
            data_frame = ttk.LabelFrame(self.root, text="Demonstrations")
            data_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
            data_frame.columnconfigure(1, weight=1)

            ttk.Label(data_frame, text="Demo directory").grid(row=0, column=0, sticky="w", padx=5, pady=4)
            ttk.Entry(data_frame, textvariable=self.demo_dir_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
            ttk.Button(
                data_frame,
                text="Browse",
                command=lambda: self._browse_directory(self.demo_dir_var, filedialog),
            ).grid(row=0, column=2, padx=5, pady=4)

            ttk.Button(
                data_frame,
                text="Load demos...",
                command=lambda: self._load_demos_via_dialog(filedialog, messagebox),
            ).grid(row=1, column=0, padx=5, pady=4, sticky="w")

            self.demo_listbox = tk.Listbox(data_frame, height=6, exportselection=False)
            self.demo_listbox.grid(row=1, column=1, sticky="nsew", padx=5, pady=4)
            scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.demo_listbox.yview)
            scrollbar.grid(row=1, column=2, sticky="ns", padx=(0, 5), pady=4)
            self.demo_listbox.configure(yscrollcommand=scrollbar.set)
            data_frame.rowconfigure(1, weight=1)

            self.demo_count_label = ttk.Label(data_frame, text="No demos selected.")
            self.demo_count_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))

            params_frame = ttk.LabelFrame(self.root, text="Model parameters")
            params_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            for idx in range(2):
                params_frame.columnconfigure(idx * 2 + 1, weight=1)

            self._add_labeled_entry(
                ttk, params_frame, "Number of GMM states (0 = auto)", self.nb_states_var, row=0, col=0
            )
            self._add_labeled_entry(ttk, params_frame, "Resampled samples (0=auto)", self.nb_samples_var, row=0, col=2)
            self._add_labeled_entry(ttk, params_frame, "Diagonal regularization", self.diag_reg_var, row=1, col=0)
            self._add_labeled_entry(ttk, params_frame, "Time scaling", self.time_scaling_var, row=1, col=2)
            self._add_labeled_entry(ttk, params_frame, "Random seed", self.random_seed_var, row=2, col=0)

            context_frame = ttk.LabelFrame(self.root, text="Context frames")
            context_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
            context_frame.columnconfigure(1, weight=1)

            ttk.Label(context_frame, text="Start frame").grid(row=0, column=0, sticky="w", padx=5, pady=4)
            self.start_combo = ttk.Combobox(context_frame, textvariable=self.start_var, state="disabled")
            self.start_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=4)

            ttk.Label(context_frame, text="Goal frame").grid(row=1, column=0, sticky="w", padx=5, pady=4)
            self.goal_combo = ttk.Combobox(context_frame, textvariable=self.goal_var, state="disabled")
            self.goal_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=4)

            ttk.Label(
                context_frame,
                text="Start frames can come from demos or from the live robot pose.\n"
                "Goal frames can come from demos or YAML files in robot_trajectories/config.",
                foreground="#444444",
            ).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))

            output_frame = ttk.LabelFrame(self.root, text="Output and visualization")
            output_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
            output_frame.columnconfigure(1, weight=1)

            ttk.Label(output_frame, text="Output directory").grid(row=0, column=0, sticky="w", padx=5, pady=4)
            ttk.Entry(output_frame, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
            ttk.Button(
                output_frame,
                text="Browse",
                command=lambda: self._browse_directory(self.output_dir_var, filedialog),
            ).grid(row=0, column=2, padx=5, pady=4)

            ttk.Label(output_frame, text="Output filename (.npz, optional)").grid(row=1, column=0, sticky="w", padx=5, pady=4)
            ttk.Entry(output_frame, textvariable=self.output_name_var).grid(row=1, column=1, sticky="ew", padx=5, pady=4)

            ttk.Label(output_frame, text="Plot file (optional PNG)").grid(row=2, column=0, sticky="w", padx=5, pady=4)
            ttk.Entry(output_frame, textvariable=self.plot_out_var).grid(row=2, column=1, sticky="ew", padx=5, pady=4)
            ttk.Button(
                output_frame,
                text="Choose file",
                command=lambda: self._browse_plot_file(filedialog),
            ).grid(row=2, column=2, padx=5, pady=4)

            self.interactive_check = ttk.Checkbutton(
                output_frame,
                text="Show interactive plot",
                variable=self.interactive_var,
                command=self._on_interactive_toggle,
            )
            self.interactive_check.grid(row=3, column=0, sticky="w", padx=5, pady=4)
            self.animation_check = ttk.Checkbutton(
                output_frame,
                text="Play trajectory animation",
                variable=self.animation_var,
            )
            self.animation_check.grid(row=3, column=1, sticky="w", padx=5, pady=4)

            action_frame = ttk.Frame(self.root)
            action_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
            action_frame.columnconfigure(0, weight=1)

            self.run_button = ttk.Button(
                action_frame,
                text="Run generalization",
                command=lambda: self._run_generalization(messagebox),
                state="disabled",
            )
            self.run_button.grid(row=0, column=0, sticky="ew", padx=5, pady=4)

            ttk.Button(
                action_frame,
                text="Load existing generalization...",
                command=lambda: self._load_existing_generalization(filedialog, messagebox),
            ).grid(row=0, column=1, sticky="ew", padx=5, pady=4)

            ttk.Button(action_frame, text="Help", command=self._show_help).grid(
                row=0, column=2, sticky="ew", padx=5, pady=4
            )

            status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
            status_bar.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))

        def _add_labeled_entry(self, ttk, parent, label_text, var, row: int, col: int):
            ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky="w", padx=5, pady=4)
            ttk.Entry(parent, textvariable=var).grid(row=row, column=col + 1, sticky="ew", padx=5, pady=4)

        def _browse_directory(self, var, filedialog):
            selected = filedialog.askdirectory(initialdir=var.get() or str(REPO_ROOT))
            if selected:
                var.set(selected)

        def _browse_plot_file(self, filedialog):
            selected = filedialog.asksaveasfilename(
                title="Save plot as...",
                defaultextension=".png",
                initialdir=self.output_dir_var.get() or str(DEFAULT_OUTPUT_DIR),
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if selected:
                self.plot_out_var.set(selected)

        def _load_demos_via_dialog(self, filedialog, messagebox):
            initial = self.demo_dir_var.get() or str(DEFAULT_DEMO_DIR)
            filenames = filedialog.askopenfilenames(
                title="Select demonstration files",
                initialdir=initial,
                filetypes=[("NumPy compressed arrays", "*.npz"), ("All files", "*.*")],
            )
            if not filenames:
                return
            self.demo_paths = [Path(name) for name in filenames]
            if not self.demo_paths:
                messagebox.showwarning("No files", "Please select at least one .npz demo file.")
                return
            self._refresh_demo_list()
            self._update_context_options()
            self._update_status(f"{len(self.demo_paths)} demo(s) selected.")

        def _refresh_demo_list(self):
            self.demo_listbox.delete(0, "end")
            for path in self.demo_paths:
                self.demo_listbox.insert("end", path.name)
            if self.demo_paths:
                self.demo_count_label.config(text=f"{len(self.demo_paths)} demo(s) selected.")
            else:
                self.demo_count_label.config(text="No demos selected.")

        def _update_context_options(self):
            if not self.demo_paths:
                self.start_combo.set("")
                self.goal_combo.set("")
                self._set_context_controls_state(False)
                return

            self.start_source_map = {}
            start_values = []
            for path in self.demo_paths:
                label = f"{path.stem} (demo start)"
                self.start_source_map[label] = ("demo", path.stem)
                start_values.append(label)
            live_label = "Real-time robot (Franka O_T_EE)"
            self.start_source_map[live_label] = ("live", None)
            start_values.append(live_label)
            self.start_combo.configure(values=start_values)
            self.start_var.set(start_values[0])

            self.goal_source_map = {}
            goal_values = []
            for path in self.demo_paths:
                label = f"{path.stem} (demo goal)"
                self.goal_source_map[label] = ("demo", path.stem)
                goal_values.append(label)
            for name, frame, cfg_path in self.config_frames:
                label = f"{name} (config)"
                self.goal_source_map[label] = ("config", frame)
                goal_values.append(label)
            if not goal_values:
                self.goal_combo.configure(values=[])
                self.goal_var.set("")
                self._set_context_controls_state(False)
                return
            self.goal_combo.configure(values=goal_values)
            self.goal_var.set(goal_values[0])
            self._set_context_controls_state(True)

        def _set_context_controls_state(self, enabled: bool):
            state = "readonly" if enabled else "disabled"
            self.start_combo.configure(state=state)
            self.goal_combo.configure(state=state)
            self.run_button.configure(state="normal" if enabled else "disabled")

        def _on_interactive_toggle(self):
            if hasattr(self, "animation_check"):
                if self.interactive_var.get():
                    self.animation_check.configure(state="normal")
                else:
                    self.animation_var.set(False)
                    self.animation_check.configure(state="disabled")

        def _int_from_var(self, var, default: int) -> int:
            try:
                return int(var.get())
            except Exception:
                return default

        def _float_from_var(self, var, default: float) -> float:
            try:
                return float(var.get())
            except Exception:
                return default

        def _load_existing_generalization(self, filedialog, messagebox):
            filename = filedialog.askopenfilename(
                title="Open generalized trajectory",
                initialdir=self.output_dir_var.get() or str(DEFAULT_OUTPUT_DIR),
                filetypes=[("NumPy compressed arrays", "*.npz"), ("All files", "*.*")],
            )
            if not filename:
                return
            try:
                _plot_saved_generalization(
                    Path(filename),
                    interactive=self.interactive_var.get(),
                    animate=self.animation_var.get(),
                )
                self._update_status(f"Loaded generalization {Path(filename).name}.")
            except Exception as exc:
                messagebox.showerror("Visualization error", f"Unable to plot {filename}:\n{exc}")
                self._update_status(f"Visualization error: {exc}")

        def _show_help(self):
            sections = [
                ("Demo directory", "Base path suggested when browsing for .npz demonstrations."),
                ("Load demos", "Pick one or more demos; they appear in the list and enable context menus."),
                ("Number of GMM states", "Set >=1 manually or enter 0 to run the automatic BIC-based selection."),
                ("Resampled samples", "Target time samples after resampling (0 keeps the reference demo length)."),
                ("Diagonal regularization", "Stabilizes covariance estimation; mirrors --diag-reg."),
                ("Time scaling", "Scales the time axis before learning for TP-GMR; mirrors --time-scaling."),
                ("Random seed", "Seed fed to TP-GMR for reproducibility."),
                ("Start frame", "Choose a demo start or pull the live robot pose from /franka_state_controller/franka_states."),
                ("Goal frame", "Pick a demo goal or any YAML pose found in robot_trajectories/config."),
                ("Output directory", "Folder where the generated generalization (.npz) will be stored."),
                ("Output filename", "Optional .npz name; leave empty to keep the timestamp-based default."),
                ("Plot file", "Optional PNG saved when visualization is requested."),
                ("Show interactive plot", "Equivalent to --interactive; required for on-screen plots/animations."),
                ("Play trajectory animation", "Equivalent to --animation; only available with the interactive plot."),
                ("Load existing generalization", "Visualize an already computed .npz without rerunning TP-GMR."),
            ]
            help_win = self._tk.Toplevel(self.root)
            help_win.title("TP-GMR GUI Help")
            help_win.geometry("520x420")
            help_win.transient(self.root)
            help_win.grab_set()
            frame = self._ttk.Frame(help_win, padding=10)
            frame.pack(fill="both", expand=True)
            text = self._tk.Text(frame, wrap="word", font=("TkDefaultFont", 10))
            scroll = self._ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            text.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            for title, desc in sections:
                text.insert("end", f"• {title}\n  {desc}\n\n")
            text.configure(state="disabled")
            close_btn = self._ttk.Button(help_win, text="Close", command=help_win.destroy)
            close_btn.pack(pady=(0, 10))

        def _update_status(self, message: str):
            self.status_var.set(message)

        def _run_generalization(self, messagebox):
            if not self.demo_paths:
                messagebox.showerror("Missing demos", "Please load at least one demonstration before running.")
                return
            try:
                params = self._gather_parameters()
                demos = _load_robot_demo_group(self.demo_paths, params["requested_samples"])
                start_frame = self._resolve_start_frame(demos)
                goal_frame = self._resolve_goal_frame(demos)
                context_frames = [start_frame, goal_frame]
                nb_states = params["nb_states"]
                if nb_states <= 0:
                    nb_states = _auto_select_nb_states(demos, params["diag_reg"], params["time_scaling"])
                result = run_tpgmr_from_demos(
                    [demo.demo for demo in demos],
                    nb_states=nb_states,
                    nb_repros=0,
                    diag_reg_factor=params["diag_reg"],
                    contexts_override=[context_frames],
                    random_state=params["random_seed"],
                    time_scaling=params["time_scaling"],
                )
                reproduction = result["reproductions"][0]
                time_scaled = result["DataIn"][0]
                output_path = _build_output_filename(params["output_dir"], params["output_name"])
                _save_generalization(
                    output_path,
                    reproduction,
                    time_scaled,
                    params["time_scaling"],
                    start_frame,
                    goal_frame,
                    demos[0].time,
                    demos[0].wrench,
                    demos[0].gripper,
                )
                if params["interactive"] or params["plot_out"] is not None or params["animate"]:
                    _plot_reproduction(
                        demos,
                        reproduction,
                        params["interactive"],
                        params["plot_out"],
                        params["animate"],
                        time=demos[0].time,
                        frames=context_frames,
                    )
                self._update_status(f"Generalization saved to {output_path}")
            except Exception as exc:
                messagebox.showerror("Generalization error", str(exc))
                self._update_status(f"Error: {exc}")

        def _gather_parameters(self):
            nb_states_raw = self._int_from_var(self.nb_states_var, 8)
            if nb_states_raw < 0:
                nb_states_raw = 0
            nb_states = nb_states_raw
            nb_samples = self._int_from_var(self.nb_samples_var, 0)
            diag_reg = max(1e-9, self._float_from_var(self.diag_reg_var, 1e-4))
            time_scaling = max(1e-6, self._float_from_var(self.time_scaling_var, 1e-1))
            random_seed = self._int_from_var(self.random_seed_var, 42)
            output_dir = Path(self.output_dir_var.get()).expanduser() if self.output_dir_var.get() else DEFAULT_OUTPUT_DIR
            plot_out = Path(self.plot_out_var.get()).expanduser() if self.plot_out_var.get().strip() else None
            output_name = self.output_name_var.get().strip() or None
            params = {
                "nb_states": nb_states,
                "requested_samples": nb_samples if nb_samples > 0 else None,
                "diag_reg": diag_reg,
                "time_scaling": time_scaling,
                "random_seed": random_seed,
                "output_dir": output_dir,
                "output_name": output_name,
                "plot_out": plot_out,
                "interactive": bool(self.interactive_var.get()),
                "animate": bool(self.animation_var.get() and self.interactive_var.get()),
            }
            return params

        def _resolve_start_frame(self, demos: Sequence[RobotDemo]) -> Frame:
            selection = self.start_var.get()
            if selection not in self.start_source_map:
                raise ValueError("Select a valid start frame option.")
            source_type, name = self.start_source_map[selection]
            if source_type == "demo":
                for demo in demos:
                    if demo.name == name:
                        return demo.frames[0]
                raise ValueError(f"Unable to find demo '{name}' for the start frame.")
            if source_type == "live":
                print("Fetching live robot pose for the start frame...")
                return _fetch_realtime_robot_frame()
            raise ValueError("Unsupported start frame selection.")

        def _resolve_goal_frame(self, demos: Sequence[RobotDemo]) -> Frame:
            selection = self.goal_var.get()
            if selection not in self.goal_source_map:
                raise ValueError("Select a valid goal frame option.")
            source_type, payload = self.goal_source_map[selection]
            if source_type == "demo":
                name = payload
                for demo in demos:
                    if demo.name == name:
                        return demo.frames[-1]
                raise ValueError(f"Unable to find demo '{name}' for the goal frame.")
            if source_type == "config":
                frame: Frame = payload
                return frame.copy()
            raise ValueError("Unsupported goal frame selection.")

    root = tk.Tk()
    RobotGeneralizationGUI(root, defaults)
    root.mainloop()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gui:
        _launch_gui(args)
        return
    demo_files = _list_demo_files(args.demo_dir)
    if not demo_files:
        raise SystemExit(f"Nessuna demo trovata in {args.demo_dir}.")
    try:
        if args.select is not None:
            selected_indices = _parse_selection(args.select, demo_files)
        else:
            selected_indices = _prompt_indices(demo_files)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    requested_samples = args.nb_samples if args.nb_samples > 0 else None
    selected_paths = [demo_files[idx] for idx in selected_indices]
    demos = _load_robot_demo_group(selected_paths, requested_samples)
    reference_demo = demos[0]
    for path in selected_paths:
        print(f"Caricata demo {path.name}")

    nb_states = args.nb_states
    if nb_states <= 0:
        nb_states = _auto_select_nb_states(demos, args.diag_reg, args.time_scaling)

    start_idx = _choose_context("di partenza", demos, 0, args.start_context)
    goal_idx = _choose_context("goal", demos, len(demos) - 1, args.goal_context)
    context_frames = [demos[start_idx].frames[0], demos[goal_idx].frames[1]]

    result = run_tpgmr_from_demos(
        [demo.demo for demo in demos],
        nb_states=nb_states,
        nb_repros=0,
        diag_reg_factor=args.diag_reg,
        contexts_override=[context_frames],
        random_state=args.random_seed,
        time_scaling=args.time_scaling,
    )
    reproduction = result["reproductions"][0]
    time_scaled = result["DataIn"][0]
    output_path = _build_output_filename(args.output_dir, args.output_name)
    _save_generalization(
        output_path,
        reproduction,
        time_scaled,
        args.time_scaling,
        context_frames[0],
        context_frames[1],
        reference_demo.time,
        reference_demo.wrench,
        reference_demo.gripper,
    )
    if args.interactive or args.plot_out is not None or args.animation:
        _plot_reproduction(
            demos,
            reproduction,
            args.interactive,
            args.plot_out,
            animate=args.animation,
            time=reference_demo.time,
            frames=context_frames,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
