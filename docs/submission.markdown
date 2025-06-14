---
layout: page
title: Submission and Evaluation
permalink: /3-submission/
weight: 3
---

## Submission
### **Model Submission Requirements**:
Please provide your solution to [TBD](). Each team can submit multiple times and we will only use the latest version you submit. Your models and scripts should be accessible and runnable. 

#### **Task I Generalizable RL for Graph Optimization**
Participants need to submit 
A well-organized GitHub repository (or zip file), containing:
- **Source Code**: Training, inference, and environment scripts.  
- **README.md**: Clear instructions for reproducing the results, environment setup, and training/inference commands.  
- **Model Checkpoint**: Trained model file and loading/inference scripts.  
- **Results File**: A CSV or JSON file reporting evaluation metrics on the test benchmark graphs.  
- **Short Report**: A concise PDF or Markdown document describing the model, generalization techniques, training setup, and key observations.


#### **Task II**
Participants need to submit



### **Paper Submission Requirements**:



## Evaluation

### **Model Evaluation**:
#### **Task I**
The performance will be assessed using the following metrics:
- **Average Cut Value**: The mean cut value over all test graphs.
- **Approximation Gap**: The relative difference between the model's output and results from Gurobi or ECO.
- **Generalization Score**: The consistency of performance across different types of graph distributions.
- **Outperformance Rate**: The percentage of test graphs where the model performs better than baseline methods.
Baseline methods:
- Gurobi: Exact solver.
- ECO-DQN: RL-based solver with physics-inspired techniques.
- Random/Greedy: Simple heuristic baselines.
All models will be tested on unseen graphs from different distributions (e.g., ER, BA, PowerLaw) to evaluate their generalization ability.


#### **Task II**
The performance will be assessed by the following metrics:



### **Paper Assessment**:
