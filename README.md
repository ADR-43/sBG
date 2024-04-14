
# Shifted Beta-Geometric Model for Customer Churn (sBG-MCMC)

## Overview
This repository contains the implementation of a fully Bayesian shifted beta-geometric (sBG) model for analyzing customer churn. This model employs a hierarchical Bayesian approach to estimate customer churn propensity and survival curves, enhancing the accuracy of customer retention predictions over traditional methods.

## Introduction
The sBG model addresses the shortcomings of traditional retention models by acknowledging the variability in customer behavior over time. By modeling customer tenure using the shifted geometric distribution and incorporating Bayesian hierarchical modeling, we gain deeper insights into customer retention and attrition dynamics.

## Methodology
- **Bayesian Interpretations**: The model utilizes Bayesian statistics to incorporate prior beliefs and observed data into a comprehensive analysis of customer behavior.
- **Data and Model**: The model is structured to handle censored data, where customers' churn events are only partially observed.
- **Sampling Techniques**: Advanced Markov Chain Monte Carlo (MCMC) methods, including Random-Walk Metropolis and Hamiltonian Monte Carlo (HMC), are used to sample from complex posterior distributions.

## Installation
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
