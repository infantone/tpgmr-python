# TPGMR Python Library

A Python implementation of Task-Parameterized Gaussian Mixture Regression (TPGMR) for robot trajectory learning and generalization.

## About

This library is based on the MATLAB implementation from [PbDlib](https://gitlab.idiap.ch/rli/pbdlib-matlab) by Sylvain Calinon and colleagues at Idiap Research Institute.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd tpgmm_pylib
pip install numpy scipy matplotlib pyyaml
```

## Usage

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

## License

This project inherits the GPL-3.0 license from the original PbDlib implementation.

PbDlib is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.

PbDlib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PbDlib. If not, see <https://www.gnu.org/licenses/>.
