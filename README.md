# Muddy Paw Planner

Muddy Paw Planner (MPP) is a planner/controller for ground robots using MPPI with a kinematic lattice planner for cost-to-go estimation.
It is designed to be concise and readable to make it easy to modify or extend.
It is written in JAX to support learning methods and GPU acceleration.

## Module Overview

- `mppi.py` — MPPI optimizer with colored-noise sampling
- `colorednoise.py` — Power-law correlated noise (white, pink, brown, etc.) for smoother sampling
- `gridmap.py` — Multi-layer grid map with nearest-neighbor and bilinear interpolation (supports x, y, heading)
- `lattice_planner.py` — Lattice-based Dijkstra planner for computing cost-to-go maps used by MPPI
- `cost.py` — Map-based navigation cost functions
