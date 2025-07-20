## RLSolver_Contest task 2 starter kit

Develop GPU-accelerated RL agents to find solutions for large Ising models. This task focuses on learning generalizable solutions across different Ising model instances.

## Installation of dependencies 

pip install numpy pandas matplotlib networkx torch numba pickle

## Folder layout

| Path | What’s inside |
|------|---------------|
| `dataset/` | **Graphs for training / evaluation**|
| `Example Env/Maxcut.py` | A *single-file*, Gym-compatible environment (`IsingEnv`) showcasing the state, action and reward design. |
| `Baseline/` | Reference solvers and helper scripts.|
