import math
import pickle

import arviz as az
import numpy as np
import torch
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from torch.distributions import Beta
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
            return torch.full((self.mean.size(0), self.mean.size(0)), float("nan"))
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
    matrix = (n / (n + 5.0)) * matrix + adjustment_factor * (
        5.0 / (n + 5.0)
    ) * torch.eye(matrix.shape[0])
    return matrix


def conditional_posterior_theta_uncensored(alpha, b, y_i):
    return Beta(alpha + 1, b + y_i - 1).sample()


def conditional_posterior_theta_censored(alpha, b, size):
    # Convert the size tuple to a torch.Size object
    sample_size = torch.Size((size,))
    return Beta(alpha, b + 7).sample(sample_size)


def H(U_func, log_q, thetas, p, M_inv, y):
    M_inv = M_inv.float()
    result = torch.matmul(M_inv, p)
    return U_func(log_q, thetas, y) + 0.5 * torch.dot(p, result)


def K(p, M_inv):
    result = torch.matmul(M_inv, p)
    return 0.5 * torch.dot(p, result)


def find_reasonable_e(U_func, log_q, thetas, M_inv, y):
    epsilon = 1
    k = 1
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    momentum = MultivariateNormal(torch.zeros(2), covariance_matrix=M)

    p = momentum.sample()
    log_q = log_q.clone()
    log_q_prime, p_prime, thetas_prime = leapfrog(
        U_func, log_q, p, thetas, epsilon, M_inv, y, init=True
    )
    log_q_prime = log_q_prime.clone().detach().requires_grad_(True)
    while torch.isinf(U_func(log_q_prime, thetas_prime, y)) or torch.isnan(
        U_func(log_q_prime, thetas_prime, y)
    ):
        k *= 0.5
        log_q_prime, p_prime, thetas_prime = leapfrog(
            U_func, log_q, p, thetas, epsilon * k, M_inv, y, init=True
        )
    epsilon *= 0.5 * k

    log_q_prime, p_prime, thetas_prime = leapfrog(
        U_func, log_q, p, thetas, epsilon, M_inv, y, init=True
    )

    H_0 = H(U_func, log_q, thetas, p, M_inv, y).item()
    H_prime = H(U_func, log_q_prime, thetas_prime, p_prime, M_inv, y).item()
    ratio = H_0 - H_prime

    alpha = 2 * float((ratio > -math.log(2))) - 1

    while alpha * ratio > -alpha * math.log(2):
        epsilon *= 2**alpha
        log_q_prime, p_prime, thetas_prime = leapfrog(
            U_func, log_q, p, thetas, epsilon, M_inv, y, init=True
        )
        H_prime = H(U_func, log_q_prime, thetas_prime, p_prime, M_inv, y).item()
        ratio = H_0 - H_prime
    return epsilon / 2


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
    u_thetas = thetas[:509]
    u_y = y[:509]

    c_thetas = thetas[509:]
    c_y = y[-1]

    log_posterior = -(
        -1000 * (torch.lgamma(alpha) + torch.lgamma(b) - torch.lgamma(alpha + b))
        + alpha * torch.sum(torch.log(u_thetas))
        + (b - 2) * torch.sum(torch.log(1 - u_thetas))
        + torch.dot(u_y, torch.log(1 - u_thetas))
        + (alpha - 1) * torch.sum(torch.log(c_thetas))
        + (b + c_y - 1) * torch.sum(torch.log(1 - c_thetas))
        + torch.log(alpha)
        + torch.log(b)
    )

    # potential = 1000 * (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)) - alpha * torch.sum(
    #     torch.log(torch.maximum(thetas, min_theta))) - beta * torch.sum(
    #     torch.log(torch.maximum((1 - thetas), min_theta))) - torch.log(alpha) - torch.log(beta)

    return log_posterior


def leapfrog(U_func, q, p, thetas, epsilon, M_inv, y, init=False):
    # Ensure q is a leaf tensor
    q = q.clone().detach().requires_grad_(True)

    thetas = thetas.clone()
    M_inv = M_inv.float()

    # Compute gradient of the potential energy U with respect to q
    U_val = U_func(q, thetas, y)
    U_val.backward()

    # Update momentum by half step
    p = p - epsilon * q.grad / 2

    y_censored_size = 1000 - 509

    # print(q.grad)
    q.grad.zero_()

    # Full step for the position
    with torch.no_grad():

        q = q + epsilon * torch.matmul(M_inv, p)

    q.requires_grad_()

    # Compute gradient of the potential energy U at the new position
    U_val = U_func(q, thetas, y)
    U_val.backward()

    # Final momentum update by half step
    p = p - epsilon * q.grad / 2

    q.grad.zero_()

    # Return the updated position and momentum
    return q.detach(), p.detach()


def build_tree(
    U_func,
    log_q,
    p,
    thetas,
    direction,
    depth,
    epsilon,
    log_q_0,
    p_0,
    M_inv,
    y,
    delta_max=1000,
):
    if depth == 0:
        log_q_prime, p_prime = leapfrog(
            U_func, log_q, p, thetas, direction * epsilon, M_inv, y
        )
        H_0 = H(U_func, log_q_0, thetas, p_0, M_inv, y)
        H_prime = H(U_func, log_q_prime, thetas, p_prime, M_inv, y)
        s_prime = (H_0 - H_prime) > -delta_max
        alpha_prime = torch.min(torch.tensor(1.0), torch.exp(H_0 - H_prime)).item()
        n_alpha_prime = 1
        log_sum_tree = H_0 - H_prime
        return (
            log_q_prime,
            p_prime,
            log_q_prime,
            p_prime,
            log_q_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
            log_sum_tree,
        )
    else:
        (
            log_q_minus,
            p_minus,
            log_q_plus,
            p_plus,
            log_q_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
            log_sum_tree,
        ) = build_tree(
            U_func,
            log_q,
            p,
            thetas,
            direction,
            depth - 1,
            epsilon,
            log_q_0,
            p_0,
            M_inv,
            y,
        )
        if s_prime:
            if direction == -1:
                (
                    log_q_minus,
                    p_minus,
                    _,
                    _,
                    log_q_double_prime,
                    s_double_prime,
                    alpha_double_prime,
                    n_alpha_double_prime,
                    log_sum_tree_prime,
                ) = build_tree(
                    U_func,
                    log_q_minus,
                    p_minus,
                    thetas,
                    direction,
                    depth - 1,
                    epsilon,
                    log_q_0,
                    p_0,
                    M_inv,
                    y,
                )
            else:
                (
                    _,
                    _,
                    log_q_plus,
                    p_plus,
                    log_q_double_prime,
                    s_double_prime,
                    alpha_double_prime,
                    n_alpha_double_prime,
                    log_sum_tree_prime,
                ) = build_tree(
                    U_func,
                    log_q_plus,
                    p_plus,
                    thetas,
                    direction,
                    depth - 1,
                    epsilon,
                    log_q_0,
                    p_0,
                    M_inv,
                    y,
                )
            if np.isnan(log_sum_tree_prime):
                log_sum_tree_prime = 0.0
            log_sum_total = np.logaddexp(log_sum_tree, log_sum_tree_prime)

            if log_sum_tree_prime > log_sum_total:
                log_q_prime = log_q_double_prime

            else:
                acceptance = np.exp(log_sum_tree_prime - log_sum_total)
                if np.random.uniform() < acceptance:
                    log_q_prime = log_q_double_prime
            log_sum_tree = log_sum_total
            alpha_prime += alpha_double_prime
            n_alpha_prime += n_alpha_double_prime
            s_prime = bool(
                s_double_prime
                and ((log_q_plus - log_q_minus) @ p_minus).item() >= 0
                and ((log_q_plus - log_q_minus) @ p_plus).item() >= 0
            )
        return (
            log_q_minus,
            p_minus,
            log_q_plus,
            p_plus,
            log_q_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
            log_sum_tree,
        )


def NUTS(U_func, log_alpha, log_beta, num_samples, m_adapt=1000):
    # torch.manual_seed(11101)

    log_q = torch.tensor(
        [log_alpha, log_beta], requires_grad=False, dtype=torch.float32
    )
    q = torch.exp(log_q)
    # arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    average_churn = arr[0].item() / 1000
    y = np.repeat(np.arange(1, len(arr) + 1), arr)

    y_uncensored = torch.tensor(y[:509])  # assuming y is a list or array
    y_censored_size = 1000 - 509
    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))
    y = torch.from_numpy(y).float()

    initial_thetas = torch.full((1000,), average_churn, dtype=torch.float32)
    initial_thetas[:509] = conditional_posterior_theta_uncensored(
        q[0].clone(), q[1].clone(), y_uncensored
    ).squeeze()
    initial_thetas[509:] = conditional_posterior_theta_censored(
        q[0].clone(), q[1].clone(), y_censored_size
    ).squeeze()
    temp = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    epsilon = find_reasonable_e(U_func, log_q, initial_thetas, temp, y)
    # epsilon = .2
    delta = 0.82
    mu = math.log(10.0 * epsilon)
    epsilon_bar = 1
    H_bar = 0
    gamma = 0.05
    t_0 = 10
    kappa = 0.75

    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    M_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    windows = [
        (275, 300),
        (300, 350),
        (350, 450),
        (450, 850),
        (850, 1050),
        (1050, 1850),
        (1850, 3500),
    ]
    current_window_index = 0  # Keep track of the current window index
    welford_cov = WelfordCovariance(log_q.shape[0])
    for i in range(num_samples):
        q = torch.exp(log_q)

        if i % 100 == 0:
            print("sample iteration number " + str(i))
            print(
                "alpha: "
                + str(torch.exp(log_q[0]).item())
                + " beta: "
                + str(torch.exp(log_q.clone()[1]).item())
            )
            print(epsilon)

        theta_samples[:509, i] = conditional_posterior_theta_uncensored(
            q[0].clone(), q[1].clone(), y_uncensored
        ).squeeze()
        theta_samples[509:, i] = conditional_posterior_theta_censored(
            q[0].clone(), q[1].clone(), y_censored_size
        ).squeeze()

        momentum = MultivariateNormal(torch.zeros(2), covariance_matrix=M)
        thetas = torch.tensor(
            theta_samples[:, i], requires_grad=False, dtype=torch.float32
        )

        p = momentum.sample()
        # Slice variable
        log_q_minus, p_minus = log_q.clone(), p.clone()
        log_q_plus, p_plus = log_q.clone(), p.clone()
        log_q_0, p_0 = log_q.clone(), p.clone()
        depth = 0
        s = bool(1)
        log_sum_tree = 0.0
        while s and depth < 10:
            v = 1 if torch.randint(0, 2, (1,)).item() == 1 else -1
            if v == -1:
                (
                    log_q_minus,
                    p_minus,
                    _,
                    _,
                    log_q_prime,
                    s_prime,
                    alpha,
                    n_alpha,
                    log_sum_tree_prime,
                ) = build_tree(
                    U_func,
                    log_q_minus,
                    p_minus,
                    thetas,
                    v,
                    depth,
                    epsilon,
                    log_q_0,
                    p_0,
                    M_inv,
                    y,
                )
            else:
                (
                    _,
                    _,
                    log_q_plus,
                    p_plus,
                    log_q_prime,
                    s_prime,
                    alpha,
                    n_alpha,
                    log_sum_tree_prime,
                ) = build_tree(
                    U_func,
                    log_q_plus,
                    p_plus,
                    thetas,
                    v,
                    depth,
                    epsilon,
                    log_q_0,
                    p_0,
                    M_inv,
                    y,
                )
            if (
                np.isnan(log_sum_tree)
                or np.isnan(log_sum_tree_prime)
                or np.isinf(log_sum_tree)
                or np.isinf(log_sum_tree_prime)
            ):
                print(
                    f"Encountered invalid value: log_sum_tree = {log_sum_tree}, log_sum_tree_prime = {log_sum_tree_prime}"
                )

            log_sum_total = np.logaddexp(log_sum_tree, log_sum_tree_prime)
            if s_prime:
                if log_sum_tree_prime > log_sum_total:
                    log_q = log_q_prime
                else:
                    acceptance = math.exp(log_sum_tree_prime - log_sum_total)
                    if np.random.uniform() < acceptance:
                        log_q = log_q_prime
            log_sum_tree = log_sum_total
            s = bool(
                s_prime
                and ((log_q_plus - log_q_minus) @ p_minus).item() >= 0
                and ((log_q_plus - log_q_minus) @ p_plus).item() >= 0
            )
            depth = depth + 1
        if i <= m_adapt:
            omega = 1 / float((i + 1 + t_0))
            H_bar = (1 - omega) * H_bar + omega * (delta - alpha / float(n_alpha))
            epsilon = math.exp(mu - (math.sqrt(i + 1) / gamma) * H_bar)
            epsilon_bar = math.exp(
                ((i + 1) ** (-kappa) * math.log(epsilon))
                + ((1 - (i + 1) ** (-kappa)) * math.log(epsilon_bar))
            )
        else:
            epsilon = epsilon_bar

        if (
            current_window_index < len(windows)
            and windows[current_window_index][0] <= i < windows[current_window_index][1]
        ):
            welford_cov.update(log_q.clone().detach().cpu())
        if i == windows[current_window_index][1] - 1:
            cov_matrix = welford_cov.covariance()
            num_samples = (
                windows[current_window_index][1] - windows[current_window_index][0]
            )
            M_inv = regularize_cov(cov_matrix, num_samples)
            M = torch.inverse(M_inv)
            print(
                f"Window {windows[current_window_index][0]}-{windows[current_window_index][1]} M_inv and M:"
            )
            print(M_inv[:2, :2])
            print(M[:2, :2])
            welford_cov = WelfordCovariance(log_q.shape[0])
            if current_window_index < len(windows) - 1:
                current_window_index += 1
        alpha_samples[0, i] = torch.exp(log_q[0]).item()
        beta_samples[0, i] = torch.exp(log_q[1]).item()
        log_q = log_q.detach().requires_grad_(False)
    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
    }

    return result


def main():
    num_samples = 5000
    num_adapt = 4000
    end = num_samples - num_adapt

    a, b, c, d = np.random.uniform(0.1, 2.25, 4)
    a_prime, b_prime, c_prime, d_prime = np.random.uniform(0.1, 10, 4)

    chain1 = NUTS(U, math.log(a), math.log(a_prime), num_samples, num_adapt)
    with open("NUTS/chain1_high_HMC.pickle", "wb") as file:
        pickle.dump(chain1, file)

    chain2 = NUTS(U, math.log(b), math.log(b_prime), num_samples, num_adapt)
    with open("NUTS/chain2_high_HMC.pickle", "wb") as file:
        pickle.dump(chain2, file)

    chain3 = NUTS(U, math.log(c), math.log(c_prime), num_samples, num_adapt)
    with open("NUTS/chain3_high_HMC.pickle", "wb") as file:
        pickle.dump(chain3, file)

    chain4 = NUTS(U, math.log(d), math.log(d_prime), num_samples, num_adapt)
    with open("NUTS/chain4_high_HMC.pickle", "wb") as file:
        pickle.dump(chain4, file)

    with open("NUTS/chain1_high_HMC.pickle", "rb") as file:
        chain1 = pickle.load(file)
    with open("NUTS/chain2_high_HMC.pickle", "rb") as file:
        chain2 = pickle.load(file)
    with open("NUTS/chain3_high_HMC.pickle", "rb") as file:
        chain3 = pickle.load(file)
    with open("NUTS/chain4_high_HMC.pickle", "rb") as file:
        chain4 = pickle.load(file)

    chains = [chain1, chain2, chain3, chain4]

    samples = []
    samples_final = []
    for chain in chains:
        sample = {"alpha": chain["alpha.samp"][0, :], "beta": chain["beta.samp"][0, :]}
        sample_final = {
            "alpha": chain["alpha.samp"][0, num_adapt:],
            "beta": chain["beta.samp"][0, num_adapt:],
        }
        samples.append(sample)
        samples_final.append(sample_final)
    # thin = int((num_samples - warm_up) * 4 / 1000)
    thin = 1

    MCMC.animate_chains(
        samples=samples,
        colors=["blue", "red", "orange", "green"],
        num_frames=2000,
        interval=20,
    )

    MCMC.animate_chains(
        samples=samples_final,
        colors=["blue", "red", "orange", "green"],
        num_frames=2000,
        interval=20,
    )

    MCMC.plot_chains(
        samples,
        parameter_name="alpha",
        colors=["blue", "red", "orange", "green"],
        begin=0,
        end=num_adapt,
        y_lim=(0, 2),
    )
    MCMC.plot_chains(
        samples_final,
        parameter_name="alpha",
        colors=["blue", "red", "orange", "green"],
        begin=0,
        end=end,
        y_lim=(0, 2),
    )

    # Plots a set of chains for the beta parameters
    MCMC.plot_chains(
        samples,
        parameter_name="beta",
        colors=["blue", "red", "orange", "green"],
        begin=0,
        end=num_adapt,
        y_lim=(0, 12),
    )
    MCMC.plot_chains(
        samples_final,
        parameter_name="beta",
        colors=["blue", "red", "orange", "green"],
        begin=0,
        end=end,
        y_lim=(0, 12),
    )

    # Warm-up removal and thinning
    final_params = {"alpha": np.array([]), "beta": np.array([])}

    for sample in samples:
        final_params["alpha"] = np.concatenate(
            (final_params["alpha"], sample["alpha"][num_adapt::thin])
        )
        final_params["beta"] = np.concatenate(
            (final_params["beta"], sample["beta"][num_adapt::thin])
        )

    print(len(final_params["alpha"]))

    az.plot_posterior(final_params["alpha"], ref_val=0.704)
    plt.show()

    az.plot_posterior(final_params["beta"], ref_val=3.86)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chain1["alpha.samp"][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chain2["alpha.samp"][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chain3["alpha.samp"][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chain4["alpha.samp"][0, num_adapt:], lags=120, ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(final_params["beta"], lags=999, ax=ax)
    plt.show()

    # Since we want to find the first lag where the autocorrelation is closest to zero before going negative,
    # we use the acf function to get the autocorrelation values directly.

    # autocorr_values = acf(final_params['alpha'], nlags=999)

    # Find the first lag where autocorrelation is closest to zero before going negative
    # Note: We start from index 1 to ignore the autocorrelation at lag 0, which is always 1.
    # for lag in range(1, len(autocorr_values), 2):  # Check only odd lags
    #     if lag + 2 < len(autocorr_values):  # Ensure we don't go out of bounds
    #         if autocorr_values[lag + 1] + autocorr_values[lag + 2] < 0:
    #             first_negative_sum_lag = lag
    #             break
    #         else:
    #             first_negative_sum_lag = 115
    #
    # # Now, we can print the identified lag and the corresponding autocorrelation values
    # print(first_negative_sum_lag)
    #
    # print(MCMC.calculate_ess(final_params['alpha'], max_lag=first_negative_sum_lag))
    #
    num_chains = len(samples_final)
    num_samples = len(
        samples_final[0]["alpha"]
    )  # Assuming all chains have the same number of samples

    # Combine samples across chains for each param

    # Create an xarray.Dataset
    data_dict = {
        "alpha": np.array([chain["alpha"] for chain in samples_final]),
        "beta": np.array([chain["beta"] for chain in samples_final]),
    }

    inference_data = az.from_dict(
        data_dict,
        coords={"chain": np.arange(num_chains), "draw": np.arange(num_samples)},
        dims={"alpha": ["chain", "draw"], "beta": ["chain", "draw"]},
    )

    ess_results = az.ess(inference_data)
    print("this is just standard ess: ", ess_results)


if __name__ == "__main__":
    main()
