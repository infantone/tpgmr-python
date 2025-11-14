#!/usr/bin/env python3
"""Sample the current Franka pose and store it under robot_trajectories/config."""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TOPIC = "/franka_state_controller/franka_states"
DEFAULT_CONTROLLER_LAUNCH = (
    "roslaunch",
    "franka_example_controllers",
    "select_controller.launch",
    "controller:=cartesian_variable_impedance_controller_passive",
)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Campiona la posa attuale del Franka leggendo %s e salva il blocco O_T_EE "
            "(e le righe successive) in un file YAML, avviando il controllore se necessario." % DEFAULT_TOPIC
        )
    )
    parser.add_argument(
        "--save",
        metavar="NOME",
        help=(
            "Nome del file (senza percorso) da creare sotto config/. Se omesso viene usato "
            "un timestamp YYYYMMDD_HHMMSS. L'estensione .yaml viene aggiunta automaticamente."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Cartella di destinazione per i file di posa (default: %(default)s).",
    )
    parser.add_argument(
        "--lines-after",
        type=int,
        default=20,
        help="Numero di righe da mantenere dopo O_T_EE nel blocco salvato (default: %(default)s).",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Topic da cui leggere lo stato del robot (default: %(default)s).",
    )
    parser.add_argument(
        "--controller-launch",
        nargs="+",
        default=list(DEFAULT_CONTROLLER_LAUNCH),
        help=(
            "Comando roslaunch per avviare il controllore se il topic non è attivo (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Tempo massimo in secondi per attendere l'attivazione del topic (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        default=True,
        help="Sovrascrive il file di destinazione se esiste già (default: attivo).",
    )
    parser.add_argument(
        "--no-force",
        action="store_false",
        dest="force",
        help="Non sovrascrive il file se esiste già.",
    )
    return parser.parse_args()


def topic_is_active(topic: str) -> bool:
    """Return True when the ROS topic exists and has at least one publisher."""

    try:
        info = subprocess.run(
            ["rostopic", "info", topic],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - ROS tooling missing locally
        raise RuntimeError("rostopic command is not available in PATH") from exc

    if info.returncode != 0:
        return False

    return "Publishers: None" not in info.stdout


def wait_for_topic(topic: str, timeout: float) -> None:
    """Block until the requested topic becomes active or we time out."""

    start = time.monotonic()
    while True:
        if topic_is_active(topic):
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Timed out after {timeout:.1f}s waiting for topic '{topic}' to become active."
            )
        time.sleep(0.5)


def capture_topic_block(topic: str, lines_after: int) -> str:
    """Return the text block that starts at O_T_EE and includes the following lines."""

    echo = subprocess.run(
        ["rostopic", "echo", "-n", "1", topic],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = echo.stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("O_T_EE:"):
            end = min(len(lines), idx + lines_after + 1)
            block = "\n".join(lines[idx:end]).strip()
            if not block:
                break
            return f"{block}\n"

    raise RuntimeError("Unable to find O_T_EE entry in the topic output.")


def terminate_process(proc: subprocess.Popen[str]) -> None:
    """Gracefully stop a roslaunch subprocess."""

    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _resolve_destination(args: argparse.Namespace) -> Path:
    config_dir = args.config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    if args.save:
        name = args.save.strip()
        if not name:
            raise RuntimeError("--save non può essere vuoto.")
        candidate = Path(name)
        if candidate.suffix != ".yaml":
            candidate = candidate.with_suffix(".yaml")
        if candidate.is_absolute():
            destination = candidate
        else:
            destination = config_dir / candidate.name
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = config_dir / f"{stamp}.yaml"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"{destination} already exists. Use --force to overwrite or specify a different --save name."
        )
    return destination


def main() -> int:
    args = parse_args()
    destination = _resolve_destination(args)

    controller_process: subprocess.Popen[str] | None = None
    controller_already_running = topic_is_active(args.topic)

    try:
        if not controller_already_running:
            print("Topic inattivo. Avvio il controllore...")
            controller_process = subprocess.Popen(args.controller_launch)
            wait_for_topic(args.topic, args.wait_timeout)
            print("Controllore attivo e topic disponibile.")

        block = capture_topic_block(args.topic, args.lines_after)
        destination.write_text(block)
        print(f"Posa salvata in {destination}")

    finally:
        if controller_process is not None:
            print("Arresto del controllore avviato dallo script...")
            terminate_process(controller_process)
            print("Controllore arrestato.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
