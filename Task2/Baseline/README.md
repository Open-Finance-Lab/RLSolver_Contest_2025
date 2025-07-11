# Baseline — MCPG & VCA for Ising

This directory ships two reference RL solvers:

| Solver | Paper | Original authors | Our fork |
|--------|-------|------------------|----------|
| **MCPG** | *Monte Carlo Policy Gradient Method for Binary Optimization * (Cheng Chen et al., 2023) | <https://github.com/optsuite/mcpg> | `src/mcpg/ising_mcpg_single_file.py` |
| **VCA** | *Variational Neural Annealing* (Mohamed Hibat-Allah et al., 2021) | <https://github.com/VectorInstitute/VariationalNeuralAnnealing> | `src/vca/ising_vna_single_file.py` |

## Quick-Train Baselines

### MCPG

- Run MCPG on on of the provided instances and provide the config yaml file:

```
python src/mcpg/ising_mcpg_single_file.py mcpg_config.yaml <instance_path>
```

### VCA

- Run VCA on on of the provided instances:

```
python src/vca/ising_vca_single_file.py <instance_path>
```

## Baseline Data

We provide 3 sets of Edwards Anderson instances from the [Variational Annealing Github](https://github.com/VectorInstitute/VariationalNeuralAnnealing): 100 spins, 400 spins, 1600 spins. Each set contains 25 instances. Additionally we provide the ground state energies per spin from the [Variational Annealing Github](https://github.com/VectorInstitute/VariationalNeuralAnnealing) and the system ground state energy calcuated from Gurobi.

System ground state energy: $$H = \sum_{i < j}J_{ij}\sigma_i\sigma_j$$

Ground state energies per spin: $$\frac{H}{\text{\# of Spins}}$$