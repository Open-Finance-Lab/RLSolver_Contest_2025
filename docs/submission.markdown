---
layout: page
title: Submission and Evaluation
permalink: /3-submission/
weight: 3
---

## Submission

### **Model Submission Requirements**:
Please provide your solution to [TBD](). Each team can submit multiple times and we will only use the latest version you submit. Your models and scripts should be accessible and runnable. 

#### **Task I MaxCut Problem**
Participants need to submit  
* A well-organized GitHub repository containing all code and a README with solution implementation instructions.  
* If applicable, a Hugging Face link to their pretrained model weights or logs.  

#### **Task II Ising Model Optimization**
Participants need to submit  
* A GitHub repository containing all scripts, models, and any custom libraries used to implement the solution.  
* A README file explaining usage, environment, and methodology.  

---

### **Paper Submission Requirements**:
Each team should submit short papers with **3 complimentary pages** and up to **2 extra pages**, including all figures, tables, and references.  
The paper submission is through the [TBD) and should follow its instructions.  
The paper title should start with:  
**“RLSolver Contest 2025 Task 1”** or **“Task 2”** depending on the task.

---

## Evaluation

For each task, the final ranking of participants will be determined by a weighted combination of **model evaluation and paper assessment**, with weights of **60%** and **40%** respectively.

### **Model Evaluation**:

#### **Task I MaxCut Problem**
The performance of the model will be assessed using the following metrics:  
1. **Objective value**: MaxCut score, higher is better.  
2. **Inference time**: Average time to solve a test instance.  
3. **Training time**: Total training time (only for RL-based methods).  
4. **Generalization**: Performance across different graph types and sizes.

#### **Task II Ising Model Optimization**
The performance of the model will be assessed using the following metrics:  
1. **Objective value**: Final energy of the Ising configuration, lower is better.  
2. **Inference time**: Time to compute the solution.  
3. **Training time**: If applicable, time spent in model training.  
4. **Scalability**: Performance across growing graph sizes.

---

### **Paper Assessment**:
The assessment of the paper will be conducted by invited experts and professionals.  
The judges will independently rate the following aspects, each accounting for **20%** of the qualitative assessment:  
* Data and model analysis  
* Robustness and generalizability  
* Innovation and creativity  
* Organization and readability  
* Technical soundness  

Note that since the paper submission will follow the timeline on the [Special Track: Reinforcement Learning for Combinatorial Optimizations(TBD) and the models can be submitted later than the paper, **the discussion of results and performance will not be counted** in the paper assessment.
