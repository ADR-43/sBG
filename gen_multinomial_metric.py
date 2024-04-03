import numpy as np
import pickle
import math
import torch
from torch.distributions import Beta
from torch.distributions.multivariate_normal import MultivariateNormal


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
    identity = torch.eye(matrix.shape[0], dtype=torch.float64)
    matrix = (n / (n + 5.0)) * matrix + adjustment_factor * (5.0 / (n + 5.0)) * identity
    return matrix


def conditional_posterior_theta_uncensored(alpha, beta, y_i):
    return Beta(torch.exp(alpha) + 1, torch.exp(beta) + y_i - 1).sample()


def conditional_posterior_theta_censored(alpha, beta, y_i):
    return Beta(torch.exp(alpha), torch.exp(beta) + y_i - 1).sample()


def H(log_q, thetas, p, M_inv, y):
    M_inv = M_inv.double()
    return U(log_q, thetas, y) + 0.5 * torch.dot(p, torch.matmul(M_inv, p))


def K(p, M_inv):
    result = torch.matmul(M_inv, p)
    return 0.5 * torch.dot(p, result)


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
    # min_theta = torch.tensor([1e-5], dtype=torch.float32)
    # alpha, beta = torch.exp(log_q).unbind()
    alpha, b = torch.exp(log_q[:2]).unbind()
    num_u = (y == torch.max(y)).nonzero()[0].item()

    u_thetas = thetas[:num_u]
    u_y = y[:num_u]

    c_thetas = thetas[num_u:]
    c_y = y[-1]

    log_posterior = - (- 1000 * (torch.lgamma(alpha) + torch.lgamma(b) - torch.lgamma(alpha + b)) + alpha * torch.sum(
        torch.log(u_thetas)) + (b - 2) * torch.sum(torch.log(1 - u_thetas)) + torch.dot(u_y, torch.log(1 - u_thetas))
                     + (alpha - 1) * torch.sum(torch.log(c_thetas)) + (b + c_y - 1) * torch.sum(
                torch.log(1 - c_thetas)) + torch.log(alpha) + torch.log(b))

    # potential = 1000 * (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)) - alpha * torch.sum(
    #     torch.log(torch.maximum(thetas, min_theta))) - beta * torch.sum(
    #     torch.log(torch.maximum((1 - thetas), min_theta))) - torch.log(alpha) - torch.log(beta)

    return log_posterior


def leapfrog(q, p, thetas, epsilon, M_inv, y):
    # Ensure q is a leaf tensor
    q = q.clone().detach().requires_grad_(True)

    thetas = thetas.clone()
    M_inv = M_inv.double()

    # Compute gradient of the potential energy U with respect to q
    U_val = U(q, thetas, y)
    U_val.backward()


    # Update momentum by half step
    p = p - epsilon * q.grad / 2

    # print(q.grad)
    q.grad.zero_()

    # Full step for the position
    with torch.no_grad():

        q = q + epsilon * torch.matmul(M_inv, p)

    q.requires_grad_()

    # Compute gradient of the potential energy U at the new position
    U_val = U(q, thetas, y)
    U_val.backward()

    # Final momentum update by half step
    p = p - epsilon * q.grad / 2


    q.grad.zero_()

    # Return the updated position and momentum
    return q.detach(), p.detach()


def build_tree(q, p, thetas, direction, depth, epsilon, q_0, p_0, M_inv, y, delta_max=1000):
    if depth == 0:
        q_prime, p_prime = leapfrog(q, p, thetas, direction * epsilon, M_inv, y)
        H_0 = H(q_0, thetas, p_0, M_inv, y)
        H_prime = H(q_prime, thetas, p_prime, M_inv, y)
        s_prime = (H_0 - H_prime) > -delta_max
        alpha_prime = torch.min(torch.tensor(1.0), torch.exp(H_0 - H_prime)).item()
        n_alpha_prime = 1
        log_sum_tree = H_0 - H_prime
        return q_prime, p_prime, q_prime, p_prime, q_prime, s_prime, alpha_prime, n_alpha_prime, log_sum_tree
    else:
        q_minus, p_minus, q_plus, p_plus, q_prime, s_prime, alpha_prime, n_alpha_prime, log_sum_tree = build_tree(q, p, thetas, direction, depth - 1, epsilon, q_0, p_0, M_inv, y)
        if s_prime:
            if direction == -1:
                q_minus, p_minus, _, _, q_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime, log_sum_tree_prime = build_tree(q_minus, p_minus, thetas, direction, depth - 1, epsilon, q_0, p_0, M_inv, y)
            else:
                _, _, q_plus, p_plus, q_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime, log_sum_tree_prime = build_tree(q_plus, p_plus, thetas, direction, depth - 1, epsilon, q_0, p_0, M_inv, y)
            if np.isnan(log_sum_tree_prime):
                log_sum_tree_prime = 0.0
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


def NUTS(data, alpha, beta, num_samples, num_adapt):
    q = torch.tensor([alpha, beta], requires_grad=False, dtype=torch.float64)

    y = np.repeat(np.arange(1, len(data) + 1), data)
    y_uncensored = torch.tensor(y[:(np.sum(data) - data[-1])], dtype=torch.float64)  # assuming y is a list or array
    y_censored = torch.tensor(y[(np.sum(data) - data[-1]):], dtype=torch.float64)  # assuming y is a list or array
    y = torch.from_numpy(y).double()

    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((np.sum(data), num_samples))

    #TODO Implement findEpsilon which involves setteng initial thetas that will be unused othwise.

    epsilon = .2
    delta = 0.82
    mu = math.log(10.0 * epsilon)
    epsilon_bar = 1
    H_bar = 0
    gamma = 0.05
    t_0 = 10
    kappa = 0.75

    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    M_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    # windows = [(275, 300), (300, 350), (350, 450), (450, 850), (850, 1050), (1050, 1850), (1850, 3500)]
    windows = [(75, 100), (100, 150), (150, 250), (250, 450), (450, 950)]
    current_window_index = 0  # Keep track of the current window index
    welford_cov = WelfordCovariance(q.shape[0])

    for i in range(num_samples):
        if i % 100 == 0:
            print('sample iteration number ' + str(i))
            print("alpha: " + str(torch.exp(q[0]).item()) + " beta: " + str(torch.exp(q[1]).item()))
            print(epsilon)
        
        
        theta_samples[:(np.sum(data) - data[-1]), i] = conditional_posterior_theta_uncensored(q[0], q[1],
                                                                        y_uncensored).squeeze()
        theta_samples[(np.sum(data) - data[-1]):, i] = conditional_posterior_theta_censored(q[0], q[1],
                                                                      y_censored).squeeze()

        momentum = MultivariateNormal(torch.zeros(2, dtype=torch.float64), covariance_matrix=M)
        thetas = torch.tensor(theta_samples[:, i], requires_grad=False, dtype=torch.float64)

        p = momentum.sample()
        q_minus, p_minus = q.clone(), p.clone()
        q_plus, p_plus = q.clone(), p.clone()
        q_0, p_0 = q.clone(), p.clone()

        depth = 0
        s = bool(1)
        log_sum_tree = 0.0
        while s and depth < 10:
            v = 1 if torch.randint(0, 2, (1,)).item() == 1 else -1
            if v == -1:
                q_minus, p_minus, _, _, q_prime, s_prime, alpha, n_alpha, log_sum_tree_prime = build_tree(q_minus, p_minus, thetas, v, depth, epsilon, q_0, p_0, M_inv, y)
            else:
                _, _, q_plus, p_plus, q_prime, s_prime, alpha, n_alpha, log_sum_tree_prime = build_tree(q_plus, p_plus, thetas, v, depth, epsilon, q_0, p_0, M_inv, y)
            if np.isnan(log_sum_tree) or np.isnan(log_sum_tree_prime) or np.isinf(log_sum_tree) or np.isinf(
                    log_sum_tree_prime):
                print(
                    f"Encountered invalid value: log_sum_tree = {log_sum_tree}, log_sum_tree_prime = {log_sum_tree_prime}")

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
            depth = depth + 1
        if i <= num_adapt:
            omega = 1 / float((i + 1 + t_0))
            H_bar = (1 - omega) * H_bar + omega * (delta - alpha / float(n_alpha))
            epsilon = math.exp(mu - (math.sqrt(i + 1) / gamma) * H_bar)
            epsilon_bar = math.exp(((i + 1) ** (-kappa) * math.log(epsilon)) + ((1 - (i + 1) ** (-kappa)) * math.log(epsilon_bar)))
        else:
            epsilon = epsilon_bar

        if current_window_index < len(windows) and windows[current_window_index][0] <= i < windows[current_window_index][1]:
            welford_cov.update(q.clone().detach().cpu())
        if i == windows[current_window_index][1] - 1:
            cov_matrix = welford_cov.covariance()
            num_samples = windows[current_window_index][1] - windows[current_window_index][0]
            M_inv = regularize_cov(cov_matrix, num_samples)
            M = torch.inverse(M_inv)
            print(f"Window {windows[current_window_index][0]}-{windows[current_window_index][1]} M_inv and M:")
            print(M_inv[:2, :2])
            print(M[:2, :2])
            welford_cov = WelfordCovariance(q.shape[0])
            if current_window_index < len(windows) - 1:
                current_window_index += 1
        alpha_samples[0, i] = torch.exp(q[0]).item()
        beta_samples[0, i] = torch.exp(q[1]).item()
        q = q.detach().requires_grad_(False)
    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
    }

    return result


def sBG(data, num_samples=2000, num_adapt=1000, chains=4):
    results = []
    for i in range(chains):
        alpha = math.log(np.random.uniform(0.25, 10))
        beta = math.log(np.random.uniform(.25, 10))

        chain = NUTS(data, alpha, beta, num_samples, num_adapt)
        with open(f'NUTS/chain{i+1}_high_HMC.pickle', 'wb') as file:
            pickle.dump(chain, file)
        
        results.append(chain)

    return results


def main():
    # This is just used for testing. Dont run this file as a script.
    num_samples = 5000
    num_adapt = 4000
    end = num_samples - num_adapt
    chains = 4

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    data = np.array([131, 126, 90, 60, 42, 34, 26, 491])

    results = sBG(data, num_samples, num_adapt, chains)

if __name__ == '__main__':
    main()
