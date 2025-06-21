---
layout: page
title: Submission and Evaluation
permalink: /submission-and-evaluation/
weight: 3
---

## Submission
### **Model Submission Requirements**:
Please provide your solution to [TBD](). Each team can submit multiple times and we will only use the latest version you submit. Your models and scripts should be accessible and runnable. 

Please submit the following three items:
1. **Neural Network Components**: Include your model definition, training script, inference script, and any trained weights or checkpoints.
2. **Code Packaging**: Provide all source files, a `requirements.txt` listing dependencies, and a brief `README.md` with setup and run instructions.
3. **Result files**: Your output should be saved in a file called `result.txt` inside the `result/` folder. Each line represents the assignment of a node to one of two sets (for MaxCut):

```
   1 2
   2 1
   3 2
   4 1
   5 2
```
- The first number is the node ID (starting from 1)  
- The second number is the assigned set (1 or 2)  

This format directly follows the example in the README. Make sure to include all nodes and follow the naming exactly.

### **Paper Submission Requirements**:
Each team should submit short papers with 3 complimentary pages and up to 2 extra pages, including all figures, tables, and references. The paper submission is through [the special track]() and should follow its instructions. Please include “RLSolver Contest Task 1/2” in your abstract.

## Evaluation

### **Model Evaluation**:
Submissions are assessed in two categories:
#### 1. Distribution-wise Reinforcement Learning Methods
- **Training Time**: total time spent training across instances  
- **Inference Time**: average time to produce a solution for a single test graph  
- **Objective Value**: primary score (e.g. MaxCut value, tour length, etc.)

#### 2. Conventional Methods
- **Running Time**: time to solve each test instance (no training phase)  
- **Objective Value**: final score achieved on the problem

| Method Type               | Time Metric              | Optimization Metric |
|---------------------------|--------------------------|---------------------|
| Distribution-wise RL      | Training + Inference     | Objective Value     |
| Conventional (non-RL)     | Running Time only        | Objective Value     |

The model ranking will be determined by TBA.

**Notes:**  
- **Objective Value** is defined by the problem (higher is better unless stated otherwise).  
- **Inference Time** is measured on GPU (when applicable) and averaged over all test cases.  
- All methods run on the same standardized server under fixed resource limits.  
- Final rankings may combine time and objective metrics with track-specific weights.  


### **Paper Assessment**:
The assessment of the paper will be conducted by invited experts and professionals. The judges will independently rate the data and model analysis, robustness and generalizability, innovation and creativity, organization and readability, each accounting for 20% of the qualitative assessment. 