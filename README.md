# RLSolver Contest 2025

This repository contains the website content and starter materials for the **RLSolver Contest 2025**.

## Outline
  - [Overview](#overview)
  - [Task 1: Graph Max-Cut with Parallel RL Agents](#task-1-graph-max-cut-with-parallel-rl-agents)
  - [Task 2: Ising Ground-State Estimation via RL-MCMC](#task-2-ising-ground-state-estimation-via-rl-mcmc)
  - [Paper Submission Requirements](#paper-submission-requirement)
  - [Resources](#resources)

## Overview

**RLSolver** explores the effectiveness of GPU-based massively parallel environments for solving large-scale combinatorial optimization (CO) problems using reinforcement learning (RL). With thousands of CUDA and tensor cores, sampling speed is improved by 2–3 orders of magnitude over CPU-based methods.

It features three main components:
- **Environments**: GPU-based simulation for CO problems
- **Agents**: RL algorithms like REINFORCE, DQN, PPO, etc.
- **Problems**: Graph Max-Cut, Ising Model, and more

We host two tasks to encourage cross-disciplinary solutions across RL, optimization, and high-performance computing.

## Task 1: Graph Max-Cut with Parallel RL Agents

Develop GPU-accelerated RL agents to solve the Max-Cut problem on large graphs. This task focuses on learning generalizable solutions across different graph distributions (e.g., BA, ER, PL).  
Starter kit coming soon.

## Task 2: Ising Ground-State Estimation via RL-MCMC

Estimate the ground state of Ising models using a reinforcement learning agent enhanced by MCMC sampling techniques.  
Starter kit coming soon.

## Paper Submission Requirements
Each team should submit short papers with 3 complimentary pages and up to 2 extra pages, including all figures, tables, and references. The paper submission is through [the special track]() and should follow its instructions. Please include “RLSolver Contest Task 1/2” in your abstract.


## Resources
[RLSolver Contest Documentation](https://rlsolver-competition.readthedocs.io/en/latest/rlsolver_contest_2025/train_test.html)

RLSolver
* [RLSolver Github Repo](https://github.com/Open-Finance-Lab/RLSolver)
* [RLSolver docs](https://rlsolvers.readthedocs.io/index.html)

RL4Ising
* [RL4Ising Github Repo](https://github.com/Open-Finance-Lab/RL4Ising)
* [RL4Ising docs](https://rl4ising-docs.readthedocs.io/en/latest/)

Relevant repositories and datasets:
* RLSolver Codebase (coming soon)
* GPU simulation environments (CUDA-based)
* Graph datasets (BA, ER, PL)
* Ising model generator and reference solvers

---

## How to Use

The website source files are in the `docs/` directory and are built using Jekyll + GitHub Pages.  
To contribute updates to CFP content or tutorials, please submit a pull request.

Website: [https://open-finance-lab.github.io/RLSolver_Contest_2025/](https://open-finance-lab.github.io/RLSolver_Contest_2025/)
