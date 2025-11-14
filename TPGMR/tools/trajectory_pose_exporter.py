#!/usr/bin/env python3
"""GUI helper to export robot poses from recorded trajectories."""

from __future__ import annotations

import ast
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
ROBOT_TRAJ_DIR = BASE_DIR / "robot_trajectories"
DEMO_DIR = ROBOT_TRAJ_DIR / "demo"
GENERALIZED_DIR = ROBOT_TRAJ_DIR / "generalized"
CONFIG_DIR = ROBOT_TRAJ_DIR / "config"

SOURCE_DIRS = {
    "Generalized trajectories": GENERALIZED_DIR,
    "Demo recordings": DEMO_DIR,
}

_PYLAB = None
_MATPLOTLIB = None


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Swap quaternion ordering from xyzw to wxyz."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("Quaternion has zero norm.")
    return quat / norm


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a quaternion (wxyz) into a 3x3 rotation matrix."""
    w, x, y, z = _quat_normalize(quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _build_transform(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Create the 4x4 transform stored in YAML configs (translation on last row)."""
    rotation = _quat_to_matrix(_quat_xyzw_to_wxyz(quat_xyzw))
    transform = np.zeros((4, 4), dtype=float)
    transform[:3, :3] = rotation
    transform[3, :3] = position
    transform[3, 3] = 1.0
    return transform


def _list_npz_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.npz"))


def _sanitize_filename(raw_name: str) -> str:
    name = raw_name.strip()
    if not name:
        return ""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return name


def _write_yaml(path: Path, matrix: np.ndarray) -> None:
    values = matrix.reshape(-1).tolist()
    serialized = ", ".join(f"{value:.16f}" for value in values)
    path.write_text(f"O_T_EE: [{serialized}]\n", encoding="utf-8")


def _get_pyplot():
    global _PYLAB, _MATPLOTLIB
    if _PYLAB is not None:
        return _PYLAB
    import matplotlib

    _MATPLOTLIB = matplotlib
    # Prefer interactive backends; fall back gracefully.
    for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "Qt4Agg", "WXAgg", "MacOSX"):
        try:
            matplotlib.use(candidate, force=True)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _PYLAB = plt
    return plt


def _load_yaml_transform(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
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
        if match:
            data = {"O_T_EE": ast.literal_eval(match.group(1))}
    if not data or "O_T_EE" not in data:
        raise ValueError(f"Unable to parse O_T_EE in {path.name}")
    transform = np.array(data["O_T_EE"], dtype=float)
    if transform.size != 16:
        raise ValueError(f"O_T_EE in {path.name} must have 16 values.")
    return transform.reshape(4, 4)


def _arrow_from_transform(ax, transform: np.ndarray, length: float = 0.1) -> None:
    origin = transform[3, :3]
    axes = transform[:3, :3] @ (np.eye(3) * length)
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


def _set_equal_axes(ax, points: list[np.ndarray]) -> None:
    if not points:
        return
    stack = np.vstack(points)
    mins = stack.min(axis=0)
    maxs = stack.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()
    span = max(span, 0.1)
    padding = span * 0.2
    for idx, axis in enumerate(("x", "y", "z")):
        low = center[idx] - span / 2 - padding
        high = center[idx] + span / 2 + padding
        getattr(ax, f"set_{axis}lim")(low, high)


def _plot_config_frames(items: list[tuple[str, np.ndarray]]) -> None:
    plt = _get_pyplot()
    fig = plt.figure("Config poses comparison", figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    origins: list[np.ndarray] = []

    for name, transform in items:
        origin = transform[3, :3]
        origins.append(origin)
        ax.scatter(origin[0], origin[1], origin[2], color="#555555", s=36)
        _arrow_from_transform(ax, transform, length=0.08)
        ax.text(origin[0], origin[1], origin[2], f" {name}", fontsize=9, color="#222222")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Selected config poses")
    ax.grid(True, alpha=0.3)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])
    _set_equal_axes(ax, origins)
    plt.show()


class PoseExporterApp:
    """Minimal Tk GUI to export start/end pose matrices from trajectories."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("TP-GMR Pose Exporter")
        self.source_var = tk.StringVar(value="Generalized trajectories")
        self.pose_var = tk.StringVar(value="start")
        self.name_var = tk.StringVar(value="custom_pose.yaml")
        self.status_var = tk.StringVar(value="Pick a trajectory to get started.")
        self.files: list[Path] = []
        self.config_files: list[Path] = []

        self._build_ui()
        self._refresh_files()
        self._refresh_config_files()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Source picker
        source_frame = ttk.LabelFrame(main, text="Trajectory source")
        source_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for idx, label in enumerate(SOURCE_DIRS):
            ttk.Radiobutton(
                source_frame,
                text=label,
                value=label,
                variable=self.source_var,
                command=self._refresh_files,
            ).grid(row=0, column=idx, padx=6, pady=6, sticky="w")

        # File list
        list_frame = ttk.LabelFrame(main, text="Available trajectories")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        main.rowconfigure(1, weight=1)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, height=12, exportselection=False)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select_file, add=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        # Pose options
        options_frame = ttk.LabelFrame(main, text="Pose export options")
        options_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            options_frame,
            text="Start pose",
            value="start",
            variable=self.pose_var,
            command=self._suggest_name,
        ).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Radiobutton(
            options_frame,
            text="Goal pose",
            value="goal",
            variable=self.pose_var,
            command=self._suggest_name,
        ).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(options_frame, text="Output filename:").grid(row=1, column=0, padx=6, pady=(0, 6), sticky="w")
        self.name_entry = ttk.Entry(options_frame, textvariable=self.name_var, width=40)
        self.name_entry.grid(row=1, column=1, padx=6, pady=(0, 6), sticky="ew")
        options_frame.columnconfigure(1, weight=1)

        # Config pose list and plotting
        config_frame = ttk.LabelFrame(main, text="Config poses")
        config_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        main.rowconfigure(3, weight=1)
        config_frame.rowconfigure(0, weight=1)
        config_frame.columnconfigure(0, weight=1)

        self.config_listbox = tk.Listbox(
            config_frame,
            height=10,
            selectmode=tk.MULTIPLE,
            exportselection=False,
        )
        self.config_listbox.grid(row=0, column=0, sticky="nsew")
        cfg_scroll = ttk.Scrollbar(config_frame, orient="vertical", command=self.config_listbox.yview)
        cfg_scroll.grid(row=0, column=1, sticky="ns")
        self.config_listbox.configure(yscrollcommand=cfg_scroll.set)

        cfg_button_frame = ttk.Frame(config_frame)
        cfg_button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        cfg_button_frame.columnconfigure(0, weight=1)
        ttk.Button(
            cfg_button_frame,
            text="Refresh config list",
            command=self._refresh_config_files,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            cfg_button_frame,
            text="Plot selected poses",
            command=self._plot_selected_configs,
        ).grid(row=0, column=1, sticky="e")

        # Actions
        action_frame = ttk.Frame(main)
        action_frame.grid(row=4, column=0, sticky="ew")
        ttk.Button(action_frame, text="Save pose to config", command=self._save_pose).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(action_frame, text="Quit", command=self.root.quit).grid(row=0, column=1)

        # Status
        status = ttk.Label(main, textvariable=self.status_var, foreground="#005a9c")
        status.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    def _refresh_files(self) -> None:
        """Reload the list of trajectories when the source changes."""
        directory = SOURCE_DIRS.get(self.source_var.get(), GENERALIZED_DIR)
        self.files = _list_npz_files(directory)
        self.listbox.delete(0, tk.END)
        for path in self.files:
            self.listbox.insert(tk.END, path.name)
        if self.files:
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")
            self.status_var.set(f"Loaded {len(self.files)} trajectories from {directory.name}.")
        else:
            self.status_var.set(f"No .npz trajectories found in {directory}.")

    def _on_select_file(self, event=None) -> None:  # noqa: ARG002
        self._suggest_name()

    def _suggest_name(self) -> None:
        """Auto-suggest an output file name based on the current selection."""
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx >= len(self.files):
            return
        pose_tag = "start" if self.pose_var.get() == "start" else "goal"
        suggested = f"{self.files[idx].stem}_{pose_tag}"
        self.name_var.set(f"{suggested}.yaml")

    def _current_path(self) -> Path | None:
        selection = self.listbox.curselection()
        if not selection:
            return None
        idx = selection[0]
        if idx >= len(self.files):
            return None
        return self.files[idx]

    def _refresh_config_files(self, highlight: str | None = None) -> None:
        if not hasattr(self, "config_listbox"):
            return
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config_files = sorted(CONFIG_DIR.glob("*.yaml"))
        self.config_listbox.delete(0, tk.END)
        highlight_idx = None
        for idx, path in enumerate(self.config_files):
            self.config_listbox.insert(tk.END, path.name)
            if highlight and path.name == highlight:
                highlight_idx = idx
        if highlight_idx is not None:
            self.config_listbox.selection_set(highlight_idx)
            self.config_listbox.see(highlight_idx)

    def _save_pose(self) -> None:
        selected_path = self._current_path()
        if selected_path is None:
            messagebox.showerror("No trajectory selected", "Select a trajectory before exporting a pose.")
            return
        sanitized_name = _sanitize_filename(self.name_var.get())
        if not sanitized_name:
            messagebox.showerror("Missing filename", "Provide a valid output filename (e.g., my_goal_pose.yaml).")
            return
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CONFIG_DIR / sanitized_name

        target_index = 0 if self.pose_var.get() == "start" else -1
        try:
            with np.load(selected_path, allow_pickle=False) as data:
                pose = np.array(data["pose"], dtype=float)
        except Exception as exc:
            messagebox.showerror("Failed to load trajectory", f"Unable to open {selected_path.name}: {exc}")
            return

        if pose.shape[0] < 7 or pose.shape[1] < 1:
            messagebox.showerror("Invalid data", f"{selected_path.name} does not contain pose samples.")
            return

        position = pose[:3, target_index]
        quaternion = pose[3:, target_index]
        try:
            transform = _build_transform(position, quaternion)
            _write_yaml(output_path, transform)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not export pose: {exc}")
            return

        self.status_var.set(f"Saved pose to {output_path.relative_to(BASE_DIR)}")
        self._refresh_config_files(highlight=output_path.name)
        messagebox.showinfo("Pose exported", f"Pose saved as {output_path.name} inside config/.")

    def _plot_selected_configs(self) -> None:
        selections = self.config_listbox.curselection()
        if not selections:
            messagebox.showerror("No config selected", "Select at least one YAML config to plot.")
            return
        items: list[tuple[str, np.ndarray]] = []
        for idx in selections:
            if idx >= len(self.config_files):
                continue
            path = self.config_files[idx]
            try:
                transform = _load_yaml_transform(path)
            except Exception as exc:
                messagebox.showerror("Failed to load config", f"{path.name}: {exc}")
                return
            items.append((path.name, transform))
        if not items:
            messagebox.showerror("No config data", "Unable to load the selected config files.")
            return
        try:
            _plot_config_frames(items)
            self.status_var.set(f"Plotted {len(items)} config pose(s).")
        except Exception as exc:  # pragma: no cover - matplotlib backend errors
            messagebox.showerror("Plot error", f"Unable to display plot: {exc}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    app = PoseExporterApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
