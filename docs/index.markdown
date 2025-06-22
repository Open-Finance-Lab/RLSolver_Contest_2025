---
layout: page
title: Overview
permalink: /
weight: 1
---

<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: center; gap: 1em; padding: 2em">
  <img style="width: 30%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/ieee-logo.png?raw=true" alt="IEEE Logo">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/columbiau.jpeg?raw=true" alt="Columbia Logo">
</div>


## Introduction

**RLSolver: GPU-based Massively Parallel Environments for Large-Scale Combinatorial Optimization (CO) Problems Using Reinforcement Learning**

RLSolver is an open-source RL-based solver for combinatorial optimization (CO) problems. RLSolver uses reinforcement learning (RL) or machine learning (ML) to automate the search process of combinatorial optimizations. It uses auto-regressive neural networks, auto-regressive graph neural networks (GNNs), or more powerful neural networks (e.g., transformer) as the policy network. With the help of GPUs with thousands of CUDA cores and tensor cores, the sampling speed is improved, which significantly enhances convergence speed and solution quality.

Two major challenges:
1.	Scalable RL/ML algorithms are highly favorable. 
2.	GPU-based simulation is the key to address the sampling bottleneck.

RLSolver consists of three key components:
- **Environments**: GPU-based massively parallel environments for CO problems.
- **Agents**: RL algorithms such as REINFORCE and DQN.
- **Problems**: Graph maxcut, TSP, Ising Model, and more.

We design **two tasks** to promote GPU-powered RL optimization:
1. **Graph Maxcut with Parallel RL Agents**
2. **Ising Model Ground-State Estimation via MCMC-based RL**

We welcome researchers, students, and practitioners from optimization, operations research (OR), RL/ML, or GPU computing communities to participate!


## Tasks

Each team can choose to participate in one or both tasks. Awards and recognitions will be given for each task.

### Task I: Graph Maxcut

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

Maximize the cut value using RL agents with multiple training environments.

---

### Task II: Ising Model

Train RL agents to find low-energy states of 2D Ising models using GPU-accelerated MCMC or spin-flip environments. The environment simulates spin lattices with interaction energy, and agents learn policies to flip spins efficiently.

#### Dataset

Generated 2D Ising model grids of various sizes (e.g., 16×16, 32×32). Participants may use or extend the provided PyTorch/CUDA simulator for custom training.

#### Goal

Minimize the total system energy across random initial configurations.

---


## Contact
Contact email: rlsolvercontest@outlook.com

Contestants can communicate any questions on 
* [Discord](https://discord.gg/QekXz9V63p).
* WeChat Group: TBA
<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: left; gap: 1em; padding: 2em">
</div>




