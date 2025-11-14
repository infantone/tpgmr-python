#!/usr/bin/env python3

"""Sample the current Franka pose and store it under robot_trajectories/config."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import yaml
from pathlib import Path

import rospy
from franka_msgs.msg import FrankaState

DEFAULT_TOPIC = "/franka_state_controller/franka_states"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Campiona la posa attuale del Franka leggendo {DEFAULT_TOPIC} e salva O_T_EE "
            "in un file YAML."
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

def main() -> int:
    args = parse_args()
    
    # Initialize ROS node
    try:
        rospy.init_node('franka_pose_sampler', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException as e:
        raise RuntimeError(
            "Impossibile inizializzare il nodo ROS. "
            "Verifica che roscore sia attivo e che tu abbia sourcato il workspace."
        ) from e
    
    destination = _resolve_destination(args)
    
    try:
        # Capture the Franka state
        state = capture_franka_state(args.topic, args.timeout)
        
        # Extract O_T_EE (16 element array representing 4x4 transformation matrix)
        o_t_ee_list = list(state.O_T_EE)
        
        # Create YAML structure with flow style (inline format)
        data = {
            'O_T_EE': o_t_ee_list
        }
        
        # Write to YAML file with flow style for the list
        with open(destination, 'w') as f:
            # Use custom representer to force flow style for lists
            yaml.dump(data, f, default_flow_style=None, width=float("inf"))
        
        # Alternative: Write manually in exact format
        with open(destination, 'w') as f:
            f.write(f"O_T_EE: {o_t_ee_list}\n")
        
        print(f"Posa salvata in {destination}")
        print(f"O_T_EE: {o_t_ee_list}")
        
        # Show position
        import numpy as np
        T = np.array(o_t_ee_list).reshape(4, 4)
        position = T[:3, 3]
        print(f"Posizione end-effector [x, y, z]: {position}")
    
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