
# Shifted Beta-Geometric Model for Customer Churn (sBG-MCMC)

## Overview
This repository contains the full Bayesian interpretation of the shifted beta-geometric (sBG) model for analyzing customer churn. This model employs a hierarchical Bayesian approach to estimate customer churn propensity and survival curves as opposed to the approxiamted version implemented using empirical Bayes (Fader and Hardie, 2007).

I originally took a Bayesian statistics class during undergrad to learn about MCMC methods which I had read about in the appendix of my stochastic processes textbook.

This project is still a work-in-progress. I work on it in my free time and started it as a way to teach myself the inner workings of HMC algorithms. It allowed me to deeply understand their derivations and restrictions. Here are some ideas I want to eventually implement.

- Generalized stopping criterion for building the trajectories using binary trees.
- An HMC algorithm generalized to Riemannian manifolds.
- More visualizations.


## Introduction
The sBG model addresses the shortcomings of traditional retention models by acknowledging the variability in customer behavior across populations (cohorts). By modeling customer tenure using the shifted geometric distribution and incorporating Bayesian hierarchical modeling, we gain deeper insights into customer retention and attrition dynamics.

## Methodology
- **Bayesian Interpretations**: The model utilizes Bayesian statistics to incorporate prior beliefs and observed data into a comprehensive analysis of customer behavior. Both empirical Bayes and fully Bayesian implementations are discussed.
- **Data and Model**: The model is structured to handle censored data, where customers' churn events are only partially observed---as would be the case in practice.
- **Sampling Techniques**: Advanced Markov Chain Monte Carlo (MCMC) methods, including Random-Walk Metropolis and Hamiltonian Monte Carlo (HMC), are used to sample from complex posterior distributions.

## Installation
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/ADR-43/sBG.git
cd sBG
```

## Use
By default, the gen.py file contains the regular and high-end customer data from the original sBG paper. To use one's own data, simply calculate the number of customers churned each period. Put that data into a numpy array with the total number of customers alive in the final period of obserdation. The length of the array should be the number of years one has been observing the activity of interest (customer churn) plus one for the censored cell of surviving customers.

| Year | Number of Customers | Number Churned |
|------|---------------------|----------------|
| 0    | 1000                | N/A            |
| 1    | 631                 | 369            |
| 2    | 438                 | 163            |
| 3    | 382                 | 86             |
| 4    | 326                 | 56             |
| 5    | 289                 | 37             |
| 6    | 262                 | 27             |
| 7    | 241                 | 21             |

Once the data is in an array named `data` within gen.py (be sure to comment out any other arrays), one can call this command from the command line/terminal.
```bash
python gen.py <number_of_samples> <number_of_chains> <sample_boolean>
```
The number of samples are the total number of samples for each chain. Commonly, there is a warm-up period for each chain to allow the algorithm to learn local (optimal step size) and global (sample parameter covariance) online. This value is set to 2000 within gen.py and can be changed if needed. The `number_of_chains` value sets the number of different chains one wants the algorithm to run. The `sample_boolean` value allows the user to set the program to create new samples (`True`) or only run the visualizations (`False`).

## File Descriptions

MCMC.py -- Implements Random-Walk Metroplis-Hastings as well as contains some utilities for visualizations. I am hoping to eventually refactor this file so that the utilities are in their own file.
NUTS(multinomial and metric).py -- A file where I test new ideas/implementations. It may be broken at any time, but I thought it would be interesting to include.
VanillaHMC.py -- Implementation of Hybrid Monte Carlo. Not generalized for any input as I have hand tuned the model parameters for the specific data. The autocorrelation is quite low; therefore, there is either currently a bug in my implementation or an excellent job has been done for parameter tuning.
gen_multinomial_metric.py -- HMC implementation generalized for the sBG. Includes multinomial sampling instead of the original slice sampling as well as an an adapted Euclidean metric for each chain.
gen.py -- Visualizations for the generalized HMC implentation for the sBG.







