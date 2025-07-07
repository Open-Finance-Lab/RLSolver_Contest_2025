# Baseline — ECO-DQN & S2V-DQN for MaxCut

This directory ships two reference RL solvers:

| Solver | Paper | Original authors | Our fork |
|--------|-------|------------------|----------|
| **ECO-DQN** | *Energy-based Combinatorial Optimisation with Deep Q-Networks* (Ben Stanton et al., 2020) | <https://github.com/benstaf/ECO-DQN> | `src/eco_dqn_*` |
| **S2V-DQN** | *Learning Combinatorial Optimization Algorithms over Graphs* (Khalil et al., ICAI 2017, arXiv 1909.04063) | <https://github.com/Hanjun-Dai/graphnn> | `src/s2v_dqn_*` |


## Quick-Test Pre-trained Baselines

### ECO-DQN

-Run the bundled checkpoint on the default test graphs:

python experiments/pre_trained_agent/test_eco.py

-To test a different model or graph set, edit experiments/pre_trained_agent/test_eco.py:
More models in the file of experiments/pre_trained_agent/networks

network_save_loc = "experiments/pre_trained_agent/networks/eco/eco_best.pth"
graph_save_loc   = "dataset/test_graphs/"

### S2V-DQN

-Run the bundled checkpoint on the default test graphs:

python experiments/pre_trained_agent/test_s2v.py

-To test a different model or graph set, edit
Inside test_s2v.py, adjust the paths:
More models in the file of experiments/pre_trained_agent/networks

network_save_loc = "experiments/pre_trained_agent/networks/s2v/s2v_best.pth"
graph_save_loc   = "dataset/test_graphs/"  # or any graph directory


## Script for Baseline data

Because the baseline models use `.pkl` files as input, we provide a simple helper script that converts a folder of GSET-style weighted edge list format `.txt` graphs into a single `.pkl` file ready for ECO-DQN and S2V-DQN.


