# TPGMR Python Library

A Python implementation of Task-Parameterized Gaussian Mixture Regression (TPGMR) for robot trajectory learning and generalization.

## About

This library is based on the MATLAB implementation from [PbDlib](https://gitlab.idiap.ch/rli/pbdlib-matlab) by Sylvain Calinon and colleagues at Idiap Research Institute.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/infantone/tpgmr-python.git
cd tpgmr-python
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy scipy matplotlib pyyaml pillow
```

## Usage

All commands should be run from the repository root directory (`tpgmr-python/`).

### Robot Trajectory Generalization

To use the library with robot trajectories:

```bash
python3 TPGMR/robot_trajectories/run_robot_generalization.py --gui
```

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
