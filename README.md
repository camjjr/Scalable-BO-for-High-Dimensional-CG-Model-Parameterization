# Scalable BO for High Dimensional CG Model Parameterization

This repository contains a Python script designed to optimize the parameters of a coarse-grained (CG) molecular dynamics (MD) model using Bayesian optimization (BO). 

Using BO, we aim to fit the CG using the following reference physical properties
1. density ($\rho$)
2. radius of gyration ($Rg$)
3. transition glass temperature ($Tg$)

The MD simulations were run using [LAMMPS](https://www.lammps.org/#gsc.tab=0), and [Optuna](https://optuna.org/) for BO.


The code automatically updates the CG's parameters (bond lengths, angles, sigmas, epsilons, etc.) in a LAMMPS input file (`parameters_search.dat`) and evaluates the quality of each set of parameters by comparing the simulation output to experimental/reference data.

## How it works

* Parameter Suggestion:
Optuna generates a new set of parameters within defined ranges (e.g., bond force constants, angle constants, Mie potential parameters).

* Parameter Injection:
The script updates the `parameters_search.dat` file by replacing placeholder tokens with trial values using the sed command.

* Simulation Execution:
Two LAMMPS simulations are run for each trial:
1. Main simulation (`input.dat`)
2. Vacuum/reference simulation (`input_vaccum.dat`)

* Objective Calculation:
The simulation outputs are parsed to compute:

- Density behavior over a temperature range (using linear regression to capture trends).
- Radius of gyration (Rg) compared to the target.

The objective function sums the squared relative errors between simulated and target values, including penalties for incorrect slopes (related to properties like the glass transition temperature, $Tg$).

* Optimization Loop:
The objective is minimized over several trials, leading to an optimized set of coarse-grained parameters.

## How to run
### Dependencies
- [Optuna](https://optuna.readthedocs.io/en/stable/installation.html)
- [LAMMPS mpirun](https://docs.lammps.org/Build_basics.html)

Prepare the following files:

* `parameters_search.dat` with placeholders (e.g., k_bond_t1t2, sigma_t1, etc.)

* Valid LAMMPS input scripts (`input.dat`, `input_vaccum.dat`)

* Run the script:

```Python
python bo_cg_search.py
```
