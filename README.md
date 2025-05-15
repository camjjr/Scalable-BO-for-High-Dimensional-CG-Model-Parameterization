# Scalable BO for High Dimensional CG Model Parameterization

This repository contains the code and data used in our study on applying Bayesian Optimization (BO) to the parameterization of a high-dimensional coarse-grained (CG) model of the Pebax-1657 copolymer. We demonstrate that BO can efficiently optimize a 41-parameter CG model to reproduce key physical properties—density, radius of gyration, and glass transition temperature—using data from atomistic simulations.

File search.py contains the main code, while get_rg.py and get_mean.py are functions used in the main file.

.
├── search.py          # Main script running the Bayesian Optimization loop
├── get_rg.py          # Function to compute radius of gyration from trajectory data
├── get_mean.py        # Helper function to compute property averages and variances
├── data/              # Input data (atomistic properties, preprocessed trajectories, etc.)
├── results/           # Output logs and optimized parameter sets
├── README.md          # Project documentation

