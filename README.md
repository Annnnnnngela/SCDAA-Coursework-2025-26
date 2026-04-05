# SCDAA-Coursework-2025-26
Coursework for Stochastic Control and Dynamic Asset Allocation

## Authors

* Hongyu Lu (Student ID: s2813450)
* Han Shen (Student ID: s2835948)
* Qi Xia (Student ID: s2881375)

Contribution:

* Hongyu Lu: 33.3%
* Han Shen: 33.3%
* Qi Xia: 33.3%

---

## Project Description

This project implements numerical methods for solving stochastic control problems, including:

* Linear Quadratic Regulator (LQR)
* Monte Carlo validation
* Supervised learning approximation
* Deep Galerkin Method (DGM)
* Policy Iteration Algorithm (PIA)

All results in the report can be reproduced using the provided code.

---

## Requirements

Python 3.10 (recommended)

Install dependencies:

pip install -r requirements.txt

Required libraries:

* numpy
* scipy
* matplotlib
* torch

---

## How to Run

Run the following scripts to reproduce results:

Exercise 1:
python src/exercise1_1.py
python src/exercise1_2.py

Example outputs are provided in:
results/exercise1/

Includes:

- Convergence plots for Monte Carlo validation
  (e.g. error vs timesteps, error vs sample size)
  
---

Exercise 2:
python src/exercise2_supervised_learning_lqr.py

Example outputs are provided in:
results/exercise1/

Includes:

- Training loss curves for value function and control networks
- Scatter plots comparing predictions and true values
- Trained neural network models (.pt files)

---

Exercise 3:
python src/exercise3_dgm_linear_pde.py

Example outputs are provided in:
results/exercise1/

Includes:

- Training loss of the Deep Galerkin Method
- Error comparison with Monte Carlo solution
- Trained DGM model

---

Exercise 4:
python src/exercise4_policy_iteration.py

Example outputs are provided in:
results/exercise1/

Includes:

- Training loss for value and policy networks
- Error vs iteration plots
- Final trained models

---

## Reproducibility

All figures and results in the report can be reproduced by running the scripts above.

- Output files are generated during execution of each script
- By default, figures are saved in the current working directory
- The `results/` folder is used to organise selected final outputs included in the report
- Code runs on CPU (no GPU required)
- Due to stochastic components, results may vary slightly between runs, but overall trends remain consistent

---
