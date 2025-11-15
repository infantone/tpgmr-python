#!/usr/bin/env python3

"""Sample the current Franka pose and store it under robot_trajectories/config.

Lo script:
- legge O_T_EE da Franka (formato column-major di libfranka),
- applica una trasformazione fissa T_FIX (rotazione ~180° + piccolo offset),
- salva il risultato in YAML in formato column-major, compatibile con il programma di terze parti.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import rospy
from franka_msgs.msg import FrankaState
import numpy as np

DEFAULT_TOPIC = "/franka_state_controller/franka_states"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Trasformazione fissa dal frame EE di Franka al frame usato dal programma di terze parti.
# Derivata direttamente confrontando le due pose che mi hai fornito.
T_FIX = np.array(
    [
        [-0.9997684741429900,  0.0157504560657616, -0.0146590250640383, -0.0001621314504932],
        [-0.0155808178789409, -0.9998111251541102, -0.0116154166465950, -0.0015730688214846],
        [-0.0148392047907793, -0.0113843282338473,  0.9998250822373237, -0.0042269500424528],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Campiona la posa attuale del Franka leggendo {DEFAULT_TOPIC} e salva O_T_EE "
            "in un file YAML (formato column-major, compatibile con il programma di terze parti)."
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
        "--topic",
        default=DEFAULT_TOPIC,
        help="Topic da cui leggere lo stato del robot (default: %(default)s).",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in secondi per attendere il messaggio (default: %(default)s).",
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


def capture_franka_state(topic: str, timeout: float) -> FrankaState:
    """Wait for and return one FrankaState message from the specified topic."""
    print(f"Attendo messaggio da {topic}...")

    try:
        msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
        print("Messaggio ricevuto!")
        return msg
    except rospy.ROSException as e:
        raise RuntimeError(
            f"Timeout dopo {timeout}s in attesa del messaggio da {topic}. "
            "Verifica che il controllore sia attivo e pubblichi sul topic."
        ) from e


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


def _franka_state_to_matrix(state: FrankaState) -> np.ndarray:
    """
    Ricostruisce la matrice 4x4 dal vettore O_T_EE pubblicato da Franka/libfranka.

    Franka usa column-major:
    [r00, r10, r20, 0,
     r01, r11, r21, 0,
     r02, r12, r22, 0,
     px,  py,  pz, 1]
    """
    raw = np.asarray(state.O_T_EE, dtype=float)
    if raw.size != 16:
        raise ValueError("O_T_EE must contain 16 elements.")
    matrix = raw.reshape(4, 4, order="F")  # column-major → matrice corretta
    return matrix


def _map_to_third_party_frame(matrix: np.ndarray) -> np.ndarray:
    """
    Applica la trasformazione fissa T_FIX per ottenere la stessa posa
    (stesso frame) del programma di terze parti:

        T_terze_parti = T_franka @ T_FIX
    """
    return matrix @ T_FIX


def _write_transform(destination: Path, matrix: np.ndarray) -> None:
    """
    Scrive O_T_EE in YAML mantenendo il formato column-major.
    """
    values = matrix.reshape(16, order="F").tolist()  # column-major
    serialized = ", ".join(f"{value:.16f}" for value in values)
    destination.write_text(f"O_T_EE: [{serialized}]\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    # Initialize ROS node
    try:
        rospy.init_node("franka_pose_sampler", anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException as e:
        raise RuntimeError(
            "Impossibile inizializzare il nodo ROS. "
            "Verifica che roscore sia attivo e che tu abbia sourcato il workspace."
        ) from e

    destination = _resolve_destination(args)

    try:
        # Capture the Franka state
        state = capture_franka_state(args.topic, args.timeout)

        # Posa nel frame di Franka
        T_franka = _franka_state_to_matrix(state)

        # Posa nel frame del programma di terze parti
        T_third = _map_to_third_party_frame(T_franka)

        _write_transform(destination, T_third)

        # Vettore O_T_EE nello stesso layout column-major che finisce nello YAML
        vec = T_third.reshape(16, order="F")
        print(f"Posa salvata in {destination}")
        print(f"O_T_EE (column-major, frame terze parti): {vec.tolist()}")

        # Posizione nel frame di terze parti
        position = T_third[:3, 3]
        print(f"Posizione end-effector [x, y, z]: {position.tolist()}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

