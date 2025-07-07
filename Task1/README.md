## RLSolver_Contest task 1 starter kit

Develop GPU-accelerated RL agents to solve the Max-Cut problem on large graphs. This task focuses on learning generalizable solutions across different graph distributions (e.g., BA, ER, PL).

## Installation of dependencies 

pip install numpy pandas matplotlib networkx torch numba pickle

## Folder layout

| Path | What’s inside |
|------|---------------|
| `dataset/` | **Graphs for training / evaluation**|
| `Example Env/Maxcut.py` | A *single-file*, Gym-compatible environment (`MaxCutEnv`) showcasing the state, action and reward design. |
| `Baseline/` | Reference solvers and helper scripts. Implementations of **ECO-DQN** and **S2V-DQN** (paper: <https://arxiv.org/abs/1909.04063>).|

## Other Method

More info and methods can be found at <https://github.com/Open-Finance-Lab/RLSolver/tree/master?tab=readme-ov-file#methods>
