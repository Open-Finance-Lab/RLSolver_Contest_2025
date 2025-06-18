---
layout: page
title: Overview
permalink: /
weight: 1
---

<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: center; gap: 1em; padding: 2em">
  <img style="width: 30%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/ieee-logo.png?raw=true" alt="IEEE Logo">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/columbiau.jpeg?raw=true" alt="Columbia Logo">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/idea.jpeg?raw=true" alt="Idea Logo">
</div>

## Media Partners 
<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: center; gap: 1em; padding: 2em">
  <!-- Wilmott Logo -->
  <a href="https://wilmott.com/" target="_blank">
      <img style="width: 80%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/Wilmott.jpg?raw=true" alt="Wilmott Logo">
  </a>

  <!-- Paris Machine Learning Logo (Same size as PyQuant News) -->
  <a href="http://parismlgroup.org/" target="_blank">
      <img style="width: 120%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/paris_machine_learning_new.png?raw=true" alt="Paris Machine Learning Logo">
  </a>
</div>

### Thanks to the AI4Finance Foundation Open-Source Community support。

#### Please find the starter kit [here](https://github.com/Open-Finance-Lab/RLSolver_Competition_2025)!

## Introduction

**RLSolver: GPU-based Massively Parallel Environments for Large-Scale Combinatorial Optimization (CO) Problems Using Reinforcement Learning**

RLSolver aims to showcase the effectiveness of GPU-based massively parallel environments for solving large-scale combinatorial optimization problems with reinforcement learning (RL). With thousands of CUDA cores and tensor cores, the sampling speed is improved by 2–3 orders of magnitude over traditional CPU-based environments, which significantly enhances convergence speed and solution quality.

RLSolver consists of three key components:
- **Environments**: GPU-based massively parallel simulation for CO problems.
- **Agents**: RL solvers like REINFORCE, DQN, etc.
- **Problems**: Graph Max-Cut, Ising Model, and more.

We design **two competition tasks** to promote GPU-powered RL optimization:
1. **Graph Max-Cut with Parallel RL Agents**
2. **Ising Model Ground-State Estimation via MCMC-based RL**

We welcome researchers, students, and practitioners from optimization, RL, or GPU computing communities to participate!


## Tasks

Each team can choose to participate in one or both tasks. Awards and recognitions will be given for each task.

### Task I: Graph Max-Cut

Develop reinforcement learning agents to solve Max-Cut problems on large graphs. Agents must be trained in a **distribution-wise** fashion across families of graphs, utilizing GPU-based environments for sampling.

#### Dataset

Synthetic graphs generated from the following distributions:
- **BA (Barabási–Albert)**
- **ER (Erdős–Rényi)**
- **PL (Power-Law)**

Each graph file follows:

```
n m           # number of nodes and edges  
u v w         # edge from node u to v with weight w  
```

#### Goal

Maximize the cut value using RL agents with batched training environments.

---

### Task II: Ising Model

Train RL agents to find low-energy states of 2D Ising models using GPU-accelerated MCMC or spin-flip environments. The environment simulates spin lattices with interaction energy, and agents learn policies to flip spins efficiently.

#### Dataset

Generated 2D Ising model grids of various sizes (e.g., 16×16, 32×32). Participants may use or extend the provided PyTorch/CUDA simulator for custom training.

#### Goal

Minimize the total system energy across random initial configurations.

---


## Contact
Contact email: rlsolvercompetition@outlook.com

Contestants can communicate any questions on 
* [Discord](https://discord.gg/RNYsEwcXVj).
* WeChat Group:
<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: left; gap: 1em; padding: 2em">
</div>




