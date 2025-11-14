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

### Capturing Poses with Real Franka Robot

To capture target poses from a real Franka robot for trajectory generalization:

**Prerequisites:**
- Running ROS environment with Franka ROS stack
- Network connection to the robot
- Virtual environment activated (if used during installation)

**Setup on Franka PC:**

```bash
# 1. Source ROS and catkin workspace
source /opt/ros/<distro>/setup.bash
source ~/catkin_ws/devel/setup.bash

# 2. Activate virtual environment (if used)
source venv/bin/activate

# 3. Navigate to repository
cd /path/to/tpgmr-python
```

**Capture a pose:**

```bash
python3 TPGMR/robot_trajectories/capture_franka_pose.py
```

This script will:
1. Check if `/franka_state_controller/franka_states` is being published
2. Launch the Franka controller if needed
3. Execute `franka_sample.py` to record the current pose
4. Extract the end-effector transformation (O_T_EE) from ROS
5. Save the pose as a timestamped YAML file in `TPGMR/robot_trajectories/config/`

**Common options:**

```bash
# Custom filename
python3 TPGMR/robot_trajectories/capture_franka_pose.py --name final_pose_lab

# Different output directory
python3 TPGMR/robot_trajectories/capture_franka_pose.py --config-dir /tmp/config

# Overwrite existing file
python3 TPGMR/robot_trajectories/capture_franka_pose.py --name my_pose --force

# Custom franka_sample.py command
python3 TPGMR/robot_trajectories/capture_franka_pose.py --sample-command "python3 /path/to/franka_sample.py"
```

The captured pose files will be automatically detected by the GUI the next time you load demonstrations.

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
