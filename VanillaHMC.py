import math
import pickle
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import digamma, polygamma, gammaln
from torch.distributions import Beta
import arviz as az
from statsmodels.graphics.tsaplots import plot_acf
import cProfile
from torch.distributions.multivariate_normal import MultivariateNormal

import MCMC


class WelfordCovariance:
    def __init__(self, size):
        self.mean = torch.zeros(size)
        self.M2 = torch.zeros((size, size))
        self.count = 0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += torch.outer(delta, (x - self.mean))

    def covariance(self):
        if self.count < 2:
            return torch.full((self.mean.size(0), self.mean.size(0)), float('nan'))
        return self.M2 / (self.count - 1)


def regularize_cov(matrix, num_samples, adjustment_factor=1e-3):
    """
    Adjusts the diagonal elements of the matrix to push them closer to 1,
    while keeping the matrix positive definite.

    Args:
    - matrix (torch.Tensor): The matrix to be adjusted.
    - target (float): The target value for diagonal elements.
    - adjustment_factor (float): The adjustment factor to apply.

    Returns:
    - torch.Tensor: The adjusted matrix.
    """
    # Compute the amount to add to diagonal elements
    n = num_samples
    matrix = (n / (n + 5.0)) * matrix + adjustment_factor * (5.0 / (n + 5.0)) * torch.eye(matrix.shape[0])
    return matrix


def conditional_posterior_theta_uncensored(alpha, b, y_i):
    return Beta(alpha + 1, b + y_i - 1).sample()


def conditional_posterior_theta_censored(alpha, b, size):
    # Convert the size tuple to a torch.Size object
    sample_size = torch.Size((size,))
    return Beta(alpha, b + 7).sample(sample_size)


def U(log_q, thetas, y):
    """
    Compute the potential energy U for given alpha, beta, and theta values.

    Parameters:
    alpha (float): Current value of alpha.
    beta (float): Current value of beta.
    theta (array): Array of theta values.

    Returns:
    float: The potential energy U.
    """
    min_theta = torch.tensor([1e-5], dtype=torch.float32)
    alpha, b = torch.exp(log_q[:2].clone()).unbind()
    u_thetas = thetas[:759].clone()
    u_y = y[:759]

    c_thetas = thetas[759:].clone()
    c_y = y[-1]

    log_posterior = - (- 1000 * (torch.lgamma(alpha) + torch.lgamma(b) - torch.lgamma(alpha + b)) + alpha * torch.sum(
        torch.log(torch.maximum(u_thetas, min_theta))) + (b - 2) * torch.sum(torch.log(1 - u_thetas)) + torch.dot(u_y, torch.log(1 - u_thetas))
                     + (alpha - 1) * torch.sum(torch.log(torch.maximum(c_thetas, min_theta))) + (b + c_y - 1) * torch.sum(
                torch.log(1 - c_thetas)) + torch.log(alpha) + torch.log(b))

    # potential = 1000 * (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)) - alpha * torch.sum(
    #     torch.log(torch.maximum(thetas, min_theta))) - beta * torch.sum(
    #     torch.log(torch.maximum((1 - thetas), min_theta))) - torch.log(alpha) - torch.log(beta)

    return log_posterior


def HMC(U_func, log_q, thetas, epsilon, L, y, M_inv, M):
    """
    Perform one iteration of the HMC algorithm.

    Parameters:
    U_func (callable): Function to compute the potential energy U.
    grad_U_func (callable): Function to compute the gradient of U.
    initial_values (array): Initial values for the parameters alpha and beta.
    theta (array): Array of theta values.
    epsilon (float): Step size for the leapfrog algorithm.
    L (int): Number of leapfrog steps.

    Returns:
    array: New values for alpha and beta after one HMC iteration.
    """
    curr_q = log_q.clone().detach().requires_grad_(True)
    log_q = log_q.clone().detach().requires_grad_(True)
    # Sample the initial momentum
    momentum = MultivariateNormal(torch.zeros(2), covariance_matrix=M)

    p = momentum.sample()
    current_p = p.clone()
    potential = U_func(log_q, thetas, y)
    potential.backward()
    p = p - (epsilon * log_q.grad / 2)
    log_q.grad.zero_()

    for _ in range(L):
        with torch.no_grad():
            log_q = log_q + epsilon * torch.matmul(M_inv, p)

        log_q.requires_grad_(True)

        # Compute potential energy and gradients for the new position
        if _ < L - 1:  # No need to compute U and gradients on the last iteration

            potential = U_func(log_q, thetas, y)
            potential.backward()
            p = p - (epsilon * log_q.grad)

            # Clear gradients after update
            log_q.grad.zero_()
            log_q = log_q.detach()

        # Negate momentum to make the proposal symmetric

    potential = U_func(log_q, thetas, y)
    potential.backward()
    p = p - (epsilon * log_q.grad / 2)
    p = -p



    log_q.grad.zero_()
    # Evaluate potential and kinetic energies at start and end of the trajectory TODO ensure dot product if vector (everywhere as well as here)
    current_U = U_func(curr_q, thetas, y).item()
    current_K = 0.5 * torch.dot(current_p, torch.matmul(M_inv, current_p)).item()
    proposed_U = potential.item()
    proposed_K = 0.5 * torch.dot(p, torch.matmul(M_inv, p)).item()

    acceptance_prob = current_U - proposed_U + current_K - proposed_K
    # Accept or reject the state at the end of the trajectory
    if math.log(torch.rand(1).item()) < acceptance_prob:
        return log_q.detach().clone(), 1
    else:
        return log_q.detach().clone(), 0


def sampler(log_alpha, log_beta, num_samples, L, epsilon):
    epsilon_bar = 0.7 * epsilon

    torch.manual_seed(11101)
    log_q = torch.tensor([log_alpha, log_beta], requires_grad=False, dtype=torch.float32)

    arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    # arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    y = np.repeat(np.arange(1, len(arr) + 1), arr)
    y = torch.from_numpy(y).float()
    y_censored_size = 241
    y_uncensored = y[:1000 - y_censored_size] # assuming y is a list or array

    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))
    num_accepted = 0

    accepted_trajectories = []
    rejected_trajectories = []

    M = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    M_inv = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    windows = [(75, 100), (100, 150), (150, 250), (250, 650), (650, 1000)]
    current_window_index = 0  # Keep track of the current window index
    welford_cov = WelfordCovariance(log_q.shape[0])
    average_churn = arr[0].item() / 1000

    theta_samples[:, 0] = torch.full((1000,), average_churn, dtype=torch.float32)

    for i in range(0, num_samples):

        q = torch.exp(log_q).clone()

        if i % 100 == 0:
            print('sample iteration number ' + str(i))
            print(num_accepted / (i + 1))
            print("alpha: " + str(q[0].item()) + " beta: " + str(q[1].item()))

        # if i % 1 == 0:
        #     print('sample iteration number ' + str(i))
        #     print(num_accepted / (i + 1))
        #     print("alpha: " + str(alpha) + " beta: " + str(b))

        # size of censored data

        # Vectorized sample generation
        theta_samples[:1000 - y_censored_size, i] = conditional_posterior_theta_uncensored(q[0], q[1],
                                                                                           y_uncensored).squeeze()
        theta_samples[1000 - y_censored_size:, i] = conditional_posterior_theta_censored(q[0], q[1],
                                                                                         y_censored_size).squeeze()

        thetas = torch.tensor(theta_samples[:, i], dtype=torch.float32)
        log_q, accepted = HMC(U, log_q, thetas, epsilon_bar, L, y, M_inv, M)

        if current_window_index < len(windows) and windows[current_window_index][0] <= i < windows[current_window_index][1]:
            welford_cov.update(log_q.clone().detach().cpu())
        if i == windows[current_window_index][1] - 1:
            cov_matrix = welford_cov.covariance()
            num_samples = windows[current_window_index][1] - windows[current_window_index][0]
            M_inv = regularize_cov(cov_matrix, num_samples)
            M = torch.inverse(M_inv)
            print(f"Window {windows[current_window_index][0]}-{windows[current_window_index][1]} M_inv and M:")
            print(M_inv[:2, :2])
            print(M[:2, :2])
            welford_cov = WelfordCovariance(log_q.shape[0])
            if current_window_index < len(windows) - 1:
                current_window_index += 1

        alpha_samples[0, i] = torch.exp(log_q[0]).item()
        beta_samples[0, i] = torch.exp(log_q[1]).item()
        num_accepted += accepted


    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
        "acceptance_rate": num_accepted / num_samples
    }

    return result

def main():
    warm_up = 1000
    end = warm_up + 2500
    num_samples = end
    start_time = time.time()
    epsilon = 0.0400200639900527
    L = 31

    # chain1 = sampler(math.log(0.3), math.log(1), num_samples, L, epsilon)
    # with open('HMC/chain1_high_HMC.pickle', 'wb') as file:
    #     pickle.dump(chain1, file)
    #
    # chain2 = sampler(math.log(2.25), math.log(10), num_samples, L, epsilon)
    # with open('HMC/chain2_high_HMC.pickle', 'wb') as file:
    #     pickle.dump(chain2, file)
    #
    # chain3 = sampler(math.log(.4), math.log(26), num_samples, L, epsilon)
    # with open('HMC/chain3_high_HMC.pickle', 'wb') as file:
    #     pickle.dump(chain3, file)
    #
    # chain4 = sampler(math.log(1.35), math.log(.4), num_samples, L, epsilon)
    # with open('HMC/chain4_high_HMC.pickle', 'wb') as file:
    #     pickle.dump(chain4, file)

    # end_time = time.time()
    # print(end_time - start_time)

    with open('HMC/chain1_high_HMC.pickle', 'rb') as file:
        chain1 = pickle.load(file)
    with open('HMC/chain2_high_HMC.pickle', 'rb') as file:
        chain2 = pickle.load(file)
    with open('HMC/chain3_high_HMC.pickle', 'rb') as file:
        chain3 = pickle.load(file)
    with open('HMC/chain4_high_HMC.pickle', 'rb') as file:
        chain4 = pickle.load(file)

    chains = [chain1, chain2, chain3, chain4]


    samples = []
    samples_final = []
    for chain in chains:
        sample = {
            'alpha': chain['alpha.samp'][0, :],
            'beta': chain['beta.samp'][0, :]
        }
        sample_final = {
            'alpha': chain['alpha.samp'][0, warm_up:],
            'beta': chain['beta.samp'][0, warm_up:]
        }
        samples.append(sample)
        samples_final.append(sample_final)



    MCMC.animate_chains(samples=samples, colors=['blue', 'red', 'orange', 'green'], num_frames=2000, interval=20)

    #
    # # Plots posterior beliefs of a customer's propensity to churn given what we observed them doing
    # high = np.array([0, 131, 257, 347, 407, 449, 483, 509])
    # for i, val in enumerate(high):
    #     final_theta_i = np.concatenate((chain1['theta.samp'][val, warm_up:], chain2['theta.samp'][val, warm_up:],
    #                                     chain3['theta.samp'][val, warm_up:], chain4['theta.samp'][val, warm_up:]))
    #     plt.hist(final_theta_i, bins=30, density=True)
    #     plt.ylim(0, 12)
    #     plt.title(("Posterior of Theta for a customer who churns at period " + str(i + 1)) * ((i + 1) % 8 != 0) + (
    #         "Theta posterior for a customer who survives through period 7") * ((i + 1) % 8 == 0))
    #     plt.xlabel("Theta Value")
    #     plt.ylabel("Density")
    #     plt.show()

    # Plots a set of chains for the alpha parameter
    MCMC.plot_chains(samples, parameter_name='alpha', colors=['blue', 'red', 'orange', 'green'], begin=0, end=end,
                     y_lim=(0, 2))
    MCMC.plot_chains(samples, parameter_name='alpha', colors=['blue', 'red', 'orange', 'green'], begin=warm_up, end=end,
                     y_lim=(0, 2))

    # Plots a set of chains for the beta parameters
    MCMC.plot_chains(samples, parameter_name='beta', colors=['blue', 'red', 'orange', 'green'], begin=0, end=end,
                     y_lim=(0, 12))
    MCMC.plot_chains(samples, parameter_name='beta', colors=['blue', 'red', 'orange', 'green'], begin=warm_up, end=end,
                     y_lim=(0, 12))

    # Warm-up removal and thinning
    final_params = {'alpha': np.array([]), 'beta': np.array([])}



    # Iterate over each sample and concatenate
    for sample in samples:
        final_params['alpha'] = np.concatenate((final_params['alpha'], sample['alpha'][warm_up:]))
        final_params['beta'] = np.concatenate((final_params['beta'], sample['beta'][warm_up:]))

    print(len(final_params['alpha']))

    # alpha_mode = MCMC.calculate_mode(final_params['alpha'])
    # b_mode = MCMC.calculate_mode(final_params['beta'])
    # # Print summary statistics of the parameters alpha and beta
    # alpha_lower, alpha_upper = np.percentile(final_params['alpha'], [2.5, 97.5])
    # print('The mean of alpha is ' + str(np.mean(final_params['alpha'])) + ' with a standard deviation of ' + str(
    #     math.sqrt(np.var(final_params['alpha']))) + ', a mode of ' + str(
    #     alpha_mode) + ', and a posterior interval (2.5% to 97.5%) of ' + str(
    #     alpha_lower) + ' to ' + str(alpha_upper))
    #
    # beta_lower, beta_upper = np.percentile(final_params['beta'], [2.5, 97.5])
    # print('The mean of beta is ' + str(np.mean(final_params['beta'])) + ' with a standard deviation of ' + str(
    #     math.sqrt(np.var(final_params['beta']))) + ', a mode of ' + str(
    #     b_mode) + ', and a posterior interval (2.5% to 97.5%) of ' + str(
    #     beta_lower) + ' to ' + str(beta_upper))

    #
    # data = np.vstack([final_params['alpha'], final_params['beta']])
    # kde = gaussian_kde(data)
    #
    # # Create a grid over which to evaluate the KDE
    # alphas_grid, betas_grid = np.meshgrid(np.linspace(0, 2, 1000),
    #                                       np.linspace(0, 10, 1000))
    # grid_coords = np.vstack([alphas_grid.ravel(), betas_grid.ravel()])
    #
    # # Evaluate the KDE on the grid
    # kde_values = kde(grid_coords).reshape(alphas_grid.shape)
    #
    # # Create the plot
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the surface
    # ax.plot_surface(alphas_grid, betas_grid, kde_values, cmap='viridis')
    #
    # # Add labels and show the plot
    # ax.set_xlabel('Alpha')
    # ax.set_ylabel('Beta')
    # ax.set_zlabel('Density')
    # plt.show()

    az.plot_posterior(final_params['alpha'], ref_val=0.688)
    plt.show()

    az.plot_posterior(final_params['beta'], ref_val=1.182)
    plt.show()

    plot_acf(final_params['alpha'], lags=250)
    plt.show()

    plot_acf(final_params['beta'], lags=250)
    plt.show()

    num_chains = len(samples_final)
    num_samples = len(samples_final[0]['alpha'])  # Assuming all chains have the same number of samples

    # Combine samples across chains for each param

    # Create an xarray.Dataset
    data_dict = {
        'alpha': np.array([chain['alpha'] for chain in samples_final]),
        'beta': np.array([chain['beta'] for chain in samples_final])
    }

    inference_data = az.from_dict(data_dict, coords={'chain': np.arange(num_chains), 'draw': np.arange(num_samples)},
                                  dims={'alpha': ['chain', 'draw'], 'beta': ['chain', 'draw']})

    ess_results = az.ess(inference_data)
    print('this is just standard ess: ', ess_results)
    print(math.sqrt(np.var(final_params['alpha'])))
    print(math.sqrt(np.var(final_params['beta'])))


if __name__ == '__main__':
    main()
