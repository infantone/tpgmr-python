# TPGMR Python Library

A Python implementation of Task-Parameterized Gaussian Mixture Regression (TPGMR) for robot trajectory learning and generalization.

## About

This library is based on the MATLAB implementation from [PbDlib](https://gitlab.idiap.ch/rli/pbdlib-matlab) by Sylvain Calinon and colleagues at Idiap Research Institute.

## Installation

Clone the repository:

```bash
git clone https://github.com/infantone/tpgmr-python.git
cd tpgmr-python
```

### Option 1: Using Virtual Environment (Recommended)

Create and activate a virtual environment to avoid installing packages system-wide:

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### Option 2: Install System-wide

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy scipy matplotlib pyyaml pillow
```

## Usage

All commands should be run from the repository root directory (`tpgmr-python/`).

**Note:** If you used a virtual environment, make sure to activate it first:
```bash
source venv/bin/activate  # On Linux/Mac
```

### Robot Trajectory Generalization

To use the library with robot trajectories:

```bash
python3 TPGMR/robot_trajectories/run_robot_generalization.py --gui
```

### Franka pose sampling

With ROS running (ROS master plus the Franka controller packages) and this project’s virtualenv already sourced:

```bash
python3 TPGMR/robot_trajectories/franka_sample.py            # writes config/<timestamp>.yaml
python3 TPGMR/robot_trajectories/franka_sample.py --save home_pose
```

The script listens to `/franka_state_controller/franka_states`, automatically launches `select_controller.launch` if the topic has no publishers, overwrites existing files by default (add `--no-force` to prevent that), and saves the `O_T_EE` block inside `TPGMR/robot_trajectories/config/`. Fresh YAMLs appear immediately inside the Start/Goal dropdowns of the GUI.

### Example Demo

To run the 3D spiral demonstration:

```bash
python3 TPGMR/examples/run_spiral_3d_demo.py
```

## Features

- Task-Parameterized Gaussian Mixture Model (TP-GMM) learning
- Gaussian Mixture Regression (GMR) for trajectory reproduction
- Multiple frame of reference support
- Robot trajectory generalization
- 3D visualization support

## Credits

This implementation is based on PbDlib (MATLAB version) by Sylvain Calinon:
- Copyright (c) 2015-2022 Idiap Research Institute, https://idiap.ch/
- Original repository: https://gitlab.idiap.ch/rli/pbdlib-matlab
