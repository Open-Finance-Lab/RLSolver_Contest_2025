# RLSolver Competition 2025

This repository contains the website content, starter materials, and track information for the **RLSolver Competition 2025**.

## Outline
  - [Overview](#overview)
  - [Task 1: Graph Max-Cut with Parallel RL Agents](#task-1-graph-max-cut-with-parallel-rl-agents)
  - [Task 2: Ising Ground-State Estimation via RL-MCMC](#task-2-ising-ground-state-estimation-via-rl-mcmc)
  - [Call for Papers: Special Track on RL for CO](#call-for-papers-special-track-on-rl-for-co)
  - [Organizers](#organizers)
  - [Resources](#resources)

## Overview

**RLSolver** explores the effectiveness of GPU-based massively parallel environments for solving large-scale combinatorial optimization (CO) problems using reinforcement learning (RL). With thousands of CUDA and tensor cores, sampling speed is improved by 2–3 orders of magnitude over CPU-based methods.

It features three main components:
- **Environments**: GPU-based simulation for CO problems
- **Agents**: RL algorithms like REINFORCE, DQN, PPO, etc.
- **Problems**: Graph Max-Cut, Ising Model, and more

We host two competition tasks to encourage cross-disciplinary solutions across RL, optimization, and high-performance computing.

## Task 1: Graph Max-Cut with Parallel RL Agents

Develop GPU-accelerated RL agents to solve the Max-Cut problem on large graphs. This task focuses on learning generalizable solutions across different graph distributions (e.g., BA, ER, PL).  
Starter kit coming soon.

## Task 2: Ising Ground-State Estimation via RL-MCMC

Estimate the ground state of Ising models using a reinforcement learning agent enhanced by MCMC sampling techniques.  
Starter kit coming soon.

## Call for Papers: Special Track on RL for CO

We invite submissions to our **Special Track: Reinforcement Learning for Combinatorial Optimization**, which welcomes original research that applies RL to CO problems. Topics include (but are not limited to):
- Graph Max-Cut, Knapsack, ILP, and Number Partitioning
- Ising models, quantum circuit search
- Vehicle Routing, UAV Path Planning, Ridesharing
- Supply Chain Management (SCM): production, inventory, logistics
- Emerging trends in Quantum AI and generative RL

The special track is designed to foster collaboration between the optimization, machine learning, and GPU computing communities.

## Organizers

See [Organizers Page](https://open-finance-lab.github.io/RLSolver_Competition_2025/organizers)  
(Organizers list and affiliations are available on the website.)

## Resources

Relevant repositories and datasets:
* RLSolver Codebase (coming soon)
* GPU simulation environments (CUDA-based)
* Graph datasets (BA, ER, PL)
* Ising model generator and reference solvers

Past related contests and resources:
* [FinRL Contest 2023](https://github.com/Open-Finance-Lab/FinRL_Contest_2023)
* [FinRL Contest 2024](https://github.com/Open-Finance-Lab/FinRL_Contest_2024)
* [FinRL-DeepSeek](https://github.com/benstaf/FinRL_DeepSeek)

---

## How to Use

The website source files are in the `docs/` directory and are built using Jekyll + GitHub Pages.  
To contribute updates to CFP content or tutorials, please submit a pull request.

Website: [https://open-finance-lab.github.io/RLSolver_Competition_2025/](https://open-finance-lab.github.io/RLSolver_Competition_2025/)
