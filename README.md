# Rocket Science Fundamentals

Interactive Jupyter notebooks on spaceflight dynamics, from numerical methods to orbit determination.
Start from [index.ipynb](index.ipynb), which links every lesson.

| Section | Topics |
|---|---|
| [Dynamic_Systems](Dynamic_Systems) | Root finding and ODE integration, state transition matrix, stability, the two-body problem as a non-linear system |
| [Observation_and_Estimation](Observation_and_Estimation) | Statistics, stochastic processes, least squares, Kalman and extended Kalman filters |
| [Astrodynamics](Astrodynamics) | Two-body problem, orbital elements, J2 perturbations, maneuvers and transfers, interplanetary design, lunar orbits, proximity operations, three-body problem |
| [Orbit_Determination](Orbit_Determination) | Force and measurement models, process noise, optical, radar and GPS tracking (examples use Ansys ODTK) |

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Open the notebooks with JupyterLab or VS Code. Plots use the `ipympl` backend (`%matplotlib widget`).

## Python modules

The notebooks import shared code from the modules in the project root, adding the root to `sys.path`
in their first code cell.

| Module | Content |
|---|---|
| `astrodynamics.py` | Constants and planetary data, two-body and CR3BP dynamics, element conversions, planetary ephemerides, Julian dates, J2 rates, Hohmann/bi-elliptic transfers |
| `lambert.py` | Gooding's Lambert solver |
| `porkchop.py` | Porkchop plots |
| `routines.py` | Plots used by the Astrodynamics notebooks |
| `algorithms.py` | Bisection, Newton, secant, Euler, RK4, RK45 |
| `dynamic_systems.py` | Examples used by the Dynamic Systems notebooks |
| `statistic.py` | Statistics, stochastic processes and least squares examples |
| `estimation.py` | Weighted least squares and Kalman filter examples |
| `resources.py` | Plots used by the Orbit Determination notebooks |
| `range_rate.py` | Light-time corrected range example |

Units are km, s and degrees unless a docstring says otherwise. Keplerian elements are ordered
`[a, e, i, RAAN, argument of periapsis, true anomaly]`.
