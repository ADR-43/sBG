import math
import pickle

import numpy as np
import torch

from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import arviz as az


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


def sample_p(M_inv):
    u = torch.randn(M_inv.shape[0])
    L = torch.linalg.cholesky(M_inv)
    return torch.cholesky_solve(u.unsqueeze(1), L).squeeze()


def find_reasonable_e(log_q, y, M_inv):
    epsilon = 1
    k = 1

    p = torch.randn(1002)

    log_q_prime, p_prime = leapfrog(log_q, p, y, epsilon, M_inv)
    log_q_prime = log_q_prime.clone().detach().requires_grad_(True)
    while torch.isinf(-posterior(log_q_prime, y)) or torch.isnan(-posterior(log_q_prime, y)):
        k *= 0.5
        log_q_prime, p_prime = leapfrog(log_q, p, y, epsilon * k, M_inv)
    epsilon *= 0.5 * k

    log_q_prime, p_prime = leapfrog(log_q, p, y, epsilon, M_inv)

    H_0 = H(log_q, p, y, M_inv).item()
    H_prime = H(log_q_prime, p_prime, y, M_inv).item()
    ratio = H_0 - H_prime

    alpha = 2 * float((ratio > -math.log(2))) - 1

    while alpha * ratio > - alpha * math.log(2):
        epsilon *= 2 ** alpha
        log_q_prime, p_prime = leapfrog(log_q, p, y, epsilon, M_inv)
        H_prime = H(log_q_prime, p_prime, y, M_inv).item()
        ratio = H_0 - H_prime
    return epsilon / 2


def posterior(q, y):
    alpha, b = torch.exp(q[:2]).unbind()
    thetas = 1 / (1 + torch.exp(-q[2:]))
    u_thetas = thetas[:509]
    u_y = y[:509]

    c_thetas = thetas[509:]
    c_y = y[-1]

    log_posterior = (- 1000 * (torch.lgamma(alpha) + torch.lgamma(b) - torch.lgamma(alpha + b)) + alpha * torch.sum(
        torch.log(u_thetas)) + (b - 2) * torch.sum(torch.log(1 - u_thetas)) + torch.dot(u_y, torch.log(1 - u_thetas))
                     + (alpha - 1) * torch.sum(torch.log(c_thetas)) + (b + c_y - 1) * torch.sum(torch.log(1 - c_thetas)))

    jacobian = torch.log(alpha) + torch.log(b) + torch.sum(torch.log(thetas) + torch.log((1 - thetas)))

    return log_posterior + jacobian


def H(q, p, y, M_inv):
    # M_inv = M_inv.float()
    result = torch.matmul(M_inv, p)
    U = - posterior(q, y)
    K = 0.5 * torch.dot(p, result)
    return U + K


def leapfrog(q, p, y, epsilon, M_inv):
    # Ensure q is a leaf tensor
    q = q.clone().detach().requires_grad_(True)
    # M_inv = M_inv.float()

    # Compute gradient of the potential energy U with respect to q
    U = -posterior(q, y)
    U.backward()

    # Update momentum by half step
    p = p - epsilon * q.grad / 2

    # print(q.grad)
    q.grad.zero_()

    # Full step for the position
    with torch.no_grad():
        q = q + epsilon * torch.matmul(M_inv, p)

    q.requires_grad_()

    # Compute gradient of the potential energy U at the new position
    U = -posterior(q, y)
    U.backward()

    # Final momentum update by half step
    p = p - epsilon * q.grad / 2

    q.grad.zero_()

    # Return the updated position and momentum
    return q.detach(), p.detach()


def build_tree(q, p, y, direction, depth, epsilon, q_0, p_0, M_inv, delta_max=1000):
    if depth == 0:
        q_prime, p_prime = leapfrog(q, p, y, direction * epsilon, M_inv)
        H_0 = H(q_0, p_0, y, M_inv)
        H_prime = H(q_prime, p_prime, y, M_inv)
        s_prime = (H_0 - H_prime) > -delta_max
        alpha_prime = torch.min(torch.tensor(1.0), torch.exp(H_0 - H_prime)).item()
        n_alpha_prime = 1
        log_sum_tree = H_0 - H_prime
        return q_prime, p_prime, q_prime, p_prime, q_prime, s_prime, alpha_prime, n_alpha_prime, log_sum_tree
    else:

        q_minus, p_minus, q_plus, p_plus, q_prime, s_prime, alpha_prime, n_alpha_prime, log_sum_tree = build_tree(
            q, p, y, direction, depth - 1, epsilon, q_0, p_0, M_inv)

        if s_prime:
            if direction == -1:
                q_minus, p_minus, _, _, q_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime, log_sum_tree_prime = build_tree(
                    q_minus, p_minus, y, direction, depth - 1, epsilon, q_0, p_0, M_inv)
            else:
                _, _, q_plus, p_plus, q_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime, log_sum_tree_prime = build_tree(
                    q_plus, p_plus, y, direction, depth - 1, epsilon, q_0, p_0, M_inv)
            log_sum_total = np.logaddexp(log_sum_tree, log_sum_tree_prime)

            if log_sum_tree_prime > log_sum_total:
                q_prime = q_double_prime
            else:
                acceptance = np.exp(log_sum_tree_prime - log_sum_total)
                if np.random.uniform() < acceptance:
                    q_prime = q_double_prime
            log_sum_tree = log_sum_total
            alpha_prime += alpha_double_prime
            n_alpha_prime += n_alpha_double_prime
            s_prime = bool(s_double_prime and ((q_plus - q_minus) @ p_minus).item() >= 0 and (
                    (q_plus - q_minus) @ p_plus).item() >= 0)
        return q_minus, p_minus, q_plus, p_plus, q_prime, s_prime, alpha_prime, n_alpha_prime, log_sum_tree


def generate_sample(q, y, epsilon, M_inv):
    p = sample_p(M_inv)
    q_minus, p_minus = q, p
    q_plus, p_plus = q, p
    q_0, p_0 = q, p
    depth = 0
    s = bool(1)
    log_sum_tree = 0.0
    while s:
        direction = 1 if torch.randint(0, 2, (1,)).item() == 1 else -1
        if direction == -1:
            q_minus, p_minus, _, _, q_prime, s_prime, alpha, n_alpha, log_sum_tree_prime = build_tree(q_minus, p_minus, y, direction, depth, epsilon, q_0, p_0, M_inv)
        else:
            _, _, q_plus, p_plus, q_prime, s_prime, alpha, n_alpha, log_sum_tree_prime = build_tree(q_plus, p_plus, y, direction, depth, epsilon, q_0, p_0, M_inv)

        log_sum_total = np.logaddexp(log_sum_tree, log_sum_tree_prime)
        if s_prime:
            if log_sum_tree_prime > log_sum_total:
                q = q_prime
            else:
                acceptance = math.exp(log_sum_tree_prime - log_sum_total)
                if np.random.uniform() < acceptance:
                    q = q_prime
        log_sum_tree = log_sum_total
        s = bool(s_prime and ((q_plus - q_minus) @ p_minus).item() >= 0 and ((q_plus - q_minus) @ p_plus).item() >= 0)
        depth += 1

    return q, alpha, n_alpha


def NUTS(log_alpha, log_beta, num_samples, m_adapt=1000):
    log_alpha = torch.tensor([log_alpha], dtype=torch.float32)  # Convert to tensor
    log_beta = torch.tensor([log_beta], dtype=torch.float32)
    # arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    average_churn = arr[0].item() / 1000
    logit_average_churn = average_churn / (1 - average_churn)
    initial_thetas = torch.full((1000,), logit_average_churn, dtype=torch.float32)
    q = torch.cat((log_alpha, log_beta, initial_thetas), dim=0)
    print(q)
    q.requires_grad_(False)
    y = np.repeat(np.arange(1, len(arr) + 1), arr)
    y = torch.from_numpy(y).float()
    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))

    M_inv = torch.eye(1002)
    epsilon = find_reasonable_e(q, y, M_inv)
    # epsilon = .03
    delta = 0.8
    mu = math.log(.75 * epsilon)
    epsilon_bar = 1
    H_bar = 0
    gamma = 0.05
    t_0 = 10
    kappa = 0.75

    windows = [(275, 475), (475, 875), (875, 1675)]
    current_window_index = 0  # Keep track of the current window index
    welford_cov = WelfordCovariance(q.shape[0])

    for i in range(num_samples):

        if i % 1 == 0:
            print('sample iteration number ' + str(i))
            print("alpha: " + str(torch.exp(q[0]).item()) + " beta: " + str(torch.exp(q[1]).item()))
            print(epsilon)
        q, alpha, n_alpha = generate_sample(q, y, epsilon, M_inv)
        q = q.detach().requires_grad_(False)
        if i <= m_adapt:
            omega = 1 / float((i + 1 + t_0))
            # print('H_bar is: ', H_bar)
            # print(omega)
            H_bar = (1 - omega) * H_bar + omega * (delta - alpha / float(n_alpha))
            # print(a / n_a)
            epsilon = math.exp(mu - (math.sqrt(i + 1) / gamma) * H_bar)
            epsilon_bar = math.exp(((i + 1) ** (-kappa) * math.log(epsilon)) + ((1 - (i + 1) ** (-kappa)) * math.log(epsilon_bar)))
            # print('H_bar is: ', H_bar)
        else:
            epsilon = epsilon_bar

        if current_window_index < len(windows) and windows[current_window_index][0] <= i < \
                windows[current_window_index][1]:
            welford_cov.update(q.clone().detach().cpu())
        if i == windows[current_window_index][1] - 1:
            cov_matrix = welford_cov.covariance()
            num_samples = windows[current_window_index][1] - windows[current_window_index][0]
            M_inv = regularize_cov(cov_matrix, num_samples)
            print(f"Window {windows[current_window_index][0]}-{windows[current_window_index][1]} M_inv and M:")
            print(M_inv[:2, :2])
            welford_cov = WelfordCovariance(q.shape[0])
            if current_window_index < len(windows) - 1:
                current_window_index += 1



        alpha_samples[0, i] = torch.exp(q[0]).item()
        beta_samples[0, i] = torch.exp(q[1]).item()
        theta_samples[:, i] = 1 / (1 + torch.exp(-q[2:]))

    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
    }
    return result

def main():
    num_samples = 3000
    num_adapt = 2000
    end = num_samples - num_adapt

    a, b, c, d = np.random.uniform(0.1, 1.5, 4)
    a_prime, b_prime, c_prime, d_prime = np.random.uniform((a, b, c, d), 4, 4)

    chain1 = NUTS(math.log(0.2), math.log(6.5), num_samples, num_adapt)
    with open('NUTS/chain1_high_HMC.pickle', 'wb') as file:
        pickle.dump(chain1, file)

    chain2 = NUTS(math.log(b), math.log(b_prime), num_samples, num_adapt)
    with open('NUTS/chain2_high_HMC.pickle', 'wb') as file:
        pickle.dump(chain2, file)

    chain3 = NUTS(math.log(c), math.log(c_prime), num_samples, num_adapt)
    with open('NUTS/chain3_high_HMC.pickle', 'wb') as file:
        pickle.dump(chain3, file)

    chain4 = NUTS(math.log(d), math.log(d_prime), num_samples, num_adapt)
    with open('NUTS/chain4_high_HMC.pickle', 'wb') as file:
        pickle.dump(chain4, file)

    with open('NUTS/chain1_high_HMC.pickle', 'rb') as file:
        chain1 = pickle.load(file)
    with open('NUTS/chain2_high_HMC.pickle', 'rb') as file:
        chain2 = pickle.load(file)
    with open('NUTS/chain3_high_HMC.pickle', 'rb') as file:
        chain3 = pickle.load(file)
    with open('NUTS/chain4_high_HMC.pickle', 'rb') as file:
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
            'alpha': chain['alpha.samp'][0, num_adapt:],
            'beta': chain['beta.samp'][0, num_adapt:]
        }
        samples.append(sample)
        samples_final.append(sample_final)
    # thin = int((num_samples - warm_up) * 4 / 1000)
    thin = 1

    MCMC.animate_chains(samples=samples, colors=['blue', 'red', 'orange', 'green'], num_frames=2000, interval=20)

    MCMC.animate_chains(samples=samples_final, colors=['blue', 'red', 'orange', 'green'], num_frames=2000, interval=20)

    MCMC.plot_chains(samples, parameter_name='alpha', colors=['blue', 'red', 'orange', 'green'], begin=0, end=num_adapt,
                     y_lim=(0, 2))
    MCMC.plot_chains(samples_final, parameter_name='alpha', colors=['blue', 'red', 'orange', 'green'], begin=0, end=end,
                     y_lim=(0, 2))


    # Plots a set of chains for the beta parameters
    MCMC.plot_chains(samples, parameter_name='beta', colors=['blue', 'red', 'orange', 'green'], begin=0, end=num_adapt,
                     y_lim=(0, 12))
    MCMC.plot_chains(samples_final, parameter_name='beta', colors=['blue', 'red', 'orange', 'green'], begin=0, end=end,
                     y_lim=(0, 12))

    # Warm-up removal and thinning
    final_params = {'alpha': np.array([]), 'beta': np.array([])}

    for sample in samples:
        final_params['alpha'] = np.concatenate((final_params['alpha'], sample['alpha'][num_adapt::thin]))
        final_params['beta'] = np.concatenate((final_params['beta'], sample['beta'][num_adapt::thin]))

    print(len(final_params['alpha']))

    az.plot_posterior(final_params['alpha'], ref_val=0.704)
    plt.show()

    az.plot_posterior(final_params['beta'], ref_val=3.86)
    plt.show()

    plot_acf(final_params['alpha'], lags=250)
    plt.show()

    plot_acf(final_params['beta'], lags=250)
    plt.show()


if __name__ == '__main__':
    main()