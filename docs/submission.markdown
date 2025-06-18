---
layout: page
title: Submission and Evaluation
permalink: /submission-and-evaluation/
weight: 3
---

# Submission and Evaluation

## Submission Requirements

Please submit the following three items:

1. **Neural Network Components**  
   Include your model definition, training script, inference script, and any trained weights or checkpoints.

2. **Code Packaging**  
   Provide all source files, a `requirements.txt` listing dependencies, and a brief `README.md` with setup and run instructions.

3. **Solution Format**  
   Supply a single text file (`result.txt`) that encodes each node’s assignment to one of the two solution sets.

## Evaluation

Submissions are assessed in two categories:

### 1. Distribution-wise Reinforcement Learning Methods
- **Training Time**: total time spent training across instances  
- **Inference Time**: average time to produce a solution for a single test graph  
- **Objective Value**: primary score (e.g. MaxCut value, tour length, etc.)

### 2. Conventional Methods
- **Running Time**: time to solve each test instance (no training phase)  
- **Objective Value**: final score achieved on the problem

| Method Type               | Time Metric              | Optimization Metric |
|---------------------------|--------------------------|---------------------|
| Distribution-wise RL      | Training + Inference     | Objective Value     |
| Conventional (non-RL)     | Running Time only        | Objective Value     |

**Notes:**  
- **Objective Value** is defined by the problem (higher is better unless stated otherwise).  
- **Inference Time** is measured on GPU (when applicable) and averaged over all test cases.  
- All methods run on the same standardized server under fixed resource limits.  
- Final rankings may combine time and objective metrics with track-specific weights.  

