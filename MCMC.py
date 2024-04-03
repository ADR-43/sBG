import math
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import arviz as az
from datetime import datetime

import numpy as np
from scipy.stats import beta, norm, uniform, gaussian_kde, geom, lognorm
from scipy.special import beta as beta_function
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Function to compute conditional posterior of alpha, beta
def log_conditional_posterior_alpha_beta(alpha, b, thetas):
    if alpha <= 0 or b <= 0:
        return -float('inf')

    return alpha * np.sum(np.log(np.maximum(thetas, 1e-308))) + b * np.sum(
        np.log(np.maximum((1 - thetas), 1e-308))) - 1000 * math.log(beta_function(alpha, b))


def log_conditional_posterior_log_alpha_beta(log_alpha, log_beta, thetas):
    return (np.exp(log_alpha) * np.sum(np.log(np.maximum(thetas, 1e-308))) +
            np.exp(log_beta) * np.sum(np.log(np.maximum((1 - thetas), 1e-308))) -
            1000 * math.log(beta_function(np.exp(log_alpha), np.exp(log_beta))) + log_alpha + log_beta)


# Function to compute conditional posterior of theta_i
def conditional_posterior_theta_uncensored(alpha, b, y_i):
    # y_i is assumed to be an array-like object
    alphas = np.full_like(y_i, alpha + 1)
    betas = b + y_i - 1
    return beta.rvs(alphas, betas)

def conditional_posterior_theta_censored(alpha, b, size):
    return beta.rvs(alpha, b + 7, size=size)


def calculate_ess(samples, max_lag=155):
    # Calculate autocorrelation using statsmodels
    autocorr = acf(samples, nlags=max_lag, fft=True)

    # Sum the autocorrelation values up to the point they become negligible
    sum_autocorr = 1 + 2 * sum(autocorr[1:])

    # Calculate ESS
    ess = len(samples) / sum_autocorr

    return ess


def calculate_mode(samples):
    kde = gaussian_kde(samples)

    # Finding a range of values around our samples to estimate the mode
    sample_range = np.linspace(min(samples), max(samples), 10000)

    # Evaluating the density function at these points
    densities = kde(sample_range)

    # Finding the mode (the sample point where the density is maximum)
    return sample_range[np.argmax(densities)]


def plot_chains(samples, parameter_name=None, colors=None, begin=None, end=None, y_lim=None):
    for idx, chain in enumerate(samples):
        plt.plot(chain[parameter_name][begin:end], color=colors[idx], linewidth=.5,
                 label=f'Chain {idx + 1}')
    plt.title(f'{parameter_name.capitalize()} Samples of Four Chains')
    plt.xlabel("Sample Index")
    plt.ylabel("Alpha Value")
    plt.ylim(y_lim)
    plt.legend()
    plt.show()


def animate_chains(samples, x_lim=(0, 3.5), y_lim=(0, 15), num_frames=2000, interval=20, fig_size=(10, 6),
                   dpi=100, colors=None, save=False, name=None):
    # Plot setup
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    lines = []

    for idx, sample in enumerate(samples):
        line = ax.plot(sample['alpha'], sample['beta'], '.', color=colors[idx], markersize=2, linewidth=1,
                       label=f'Chain {idx + 1}')[0]
        lines.append(line)

    ax.legend()

    def init():
        for line, sample in zip(lines, samples):
            line.set_data([], [])
        return lines

    def animate(k):
        for line, sample in zip(lines, samples):
            line.set_data(sample['alpha'][:k], sample['beta'][:k])
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=interval, blit=True)

    if save:
        if not isinstance(name, str):
            raise TypeError("The 'name' argument must be a string.")
        anim.save('convergence.mp4', writer='ffmpeg', fps=30, dpi=dpi)

    plt.show()


def independent_log_conditional_posterior_log_alpha(log_alpha, log_beta, thetas):
    return np.exp(log_alpha) * np.sum(np.log(np.maximum(thetas, 1e-308))) - 1000 * math.log(
        beta_function(np.exp(log_alpha), np.exp(log_beta))) - log_alpha


def independent_log_conditional_posterior_log_beta(log_alpha, log_beta, thetas):
    return np.exp(log_beta) * np.sum(np.log(np.maximum((1 - thetas), 1e-308))) - 1000 * math.log(
        beta_function(np.exp(log_alpha), np.exp(log_beta))) - log_beta


def independent_log_gibbs(log_alpha, log_beta, num_samples, sd_a_prop, sd_b_prop):
    random.seed(a=11101)

    # arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    y = np.repeat(np.arange(1, len(arr) + 1), arr)

    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))
    num_accepted_alpha = 0
    num_accepted_beta = 0

    for i in range(num_samples):

        alpha = np.exp(log_alpha)
        b = np.exp(log_beta)

        if i % 100 == 0:
            print('sample iteration number ' + str(i))
            print(num_accepted_alpha / (i + 1))
            print(num_accepted_beta / (i + 1))
            print("alpha: " + str(alpha) + " beta: " + str(b))

        for j in range(0, 509):
            theta_samples[j, i] = conditional_posterior_theta_uncensored(alpha, b, y[j])[0]

        for j in range(509, 1000):
            theta_samples[j, i] = conditional_posterior_theta_censored(alpha, b)[0]

        log_alpha_prop = norm.rvs(loc=log_alpha, scale=sd_a_prop)
        log_beta_prop = norm.rvs(loc=log_beta, scale=sd_b_prop)
        alpha_prop = np.exp(log_alpha_prop)
        beta_prop = np.exp(log_beta_prop)

        target_prop_alpha = independent_log_conditional_posterior_log_alpha(log_alpha_prop, log_beta_prop,
                                                                            theta_samples[:, i])
        current_prop_alpha = independent_log_conditional_posterior_log_alpha(log_alpha, log_beta, theta_samples[:, i])

        target_prop_beta = independent_log_conditional_posterior_log_beta(log_alpha_prop, log_beta_prop,
                                                                          theta_samples[:, i])
        current_prop_beta = independent_log_conditional_posterior_log_beta(log_alpha, log_beta, theta_samples[:, i])

        log_given_current_alpha = lognorm.logpdf(alpha_prop, s=sd_a_prop, scale=alpha)
        log_given_current_beta = lognorm.logpdf(beta_prop, s=sd_b_prop, scale=b)

        log_given_proposed_alpha = lognorm.logpdf(alpha, s=sd_a_prop, scale=alpha_prop)
        log_given_proposed_beta = lognorm.logpdf(alpha, s=sd_a_prop, scale=alpha_prop)

        r_alpha = target_prop_alpha - current_prop_alpha + log_given_proposed_alpha - log_given_current_alpha
        r_beta = target_prop_beta - current_prop_beta + log_given_proposed_beta - log_given_current_beta

        u = np.log(uniform.rvs(loc=0, scale=1))

        if r_alpha >= u:
            log_alpha = log_alpha_prop
            num_accepted_alpha += 1

        if r_beta >= u:
            log_beta = log_beta_prop
            num_accepted_beta += 1

            # Back-transform log_alpha and log_beta to store them
        alpha_samples[0, i] = alpha_prop
        beta_samples[0, i] = beta_prop

    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
    }

    return result


def log_gibbs(log_alpha, log_beta, num_samples, sd_a_prop, sd_b_prop):
    random.seed(a=11101)

    # arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    y = np.repeat(np.arange(1, len(arr) + 1), arr)

    y_uncensored = y[:509]  # assuming y is a list or array
    y_censored_size = 1000 - 509

    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))
    num_accepted = 0

    for i in range(num_samples):

        alpha = np.exp(log_alpha)
        b = np.exp(log_beta)

        if i % 100 == 0:
            print('sample iteration number ' + str(i))
            print(num_accepted / (i + 1))
            print("alpha: " + str(alpha) + " beta: " + str(b))

        theta_samples[:509, i] = conditional_posterior_theta_uncensored(alpha, b, y_uncensored)
        theta_samples[509:, i] = conditional_posterior_theta_censored(alpha, b, y_censored_size)

        log_alpha_prop = norm.rvs(loc=log_alpha, scale=sd_a_prop)
        log_beta_prop = norm.rvs(loc=log_beta, scale=sd_b_prop)
        alpha_prop = np.exp(log_alpha_prop)
        beta_prop = np.exp(log_beta_prop)

        target_prop = log_conditional_posterior_log_alpha_beta(log_alpha_prop, log_beta_prop, theta_samples[:, i])
        current_prop = log_conditional_posterior_log_alpha_beta(log_alpha, log_beta, theta_samples[:, i])

        log_given_current = lognorm.logpdf(alpha_prop, s=sd_a_prop, scale=alpha) + lognorm.logpdf(beta_prop,
                                                                                                  s=sd_b_prop, scale=b)

        log_given_proposed = lognorm.logpdf(alpha, s=sd_a_prop, scale=alpha_prop) + lognorm.logpdf(b, s=sd_b_prop,
                                                                                                   scale=beta_prop)

        r = target_prop - current_prop + log_given_proposed - log_given_current

        u = np.log(uniform.rvs(loc=0, scale=1))

        if r >= u:
            log_alpha = log_alpha_prop
            log_beta = log_beta_prop
            num_accepted += 1

            # Back-transform log_alpha and log_beta to store them
        alpha_samples[0, i] = alpha_prop
        beta_samples[0, i] = beta_prop

    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
        "acceptance_rate": num_accepted / num_samples
    }

    return result


def gibbs(alpha, b, num_samples, sd_a_prop, sd_b_prop):
    random.seed(a=11101)

    # arr = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    arr = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    y = np.repeat(np.arange(1, len(arr) + 1), arr)

    alpha_samples = np.zeros((1, num_samples))
    beta_samples = np.zeros((1, num_samples))
    theta_samples = np.zeros((1000, num_samples))
    num_accepted = 0

    for i in range(0, num_samples):
        if i % 100 == 0:
            print('sample iteration number ' + str(i))
            print(num_accepted / (i + 1))
            print("alpha: " + str(alpha) + " beta: " + str(b))

        for j in range(0, 509):
            theta_samples[j, i] = conditional_posterior_theta_uncensored(alpha, b, y[j])[0]

        for j in range(509, 1000):
            theta_samples[j, i] = conditional_posterior_theta_censored(alpha, b)[0]

        alpha_prop = norm.rvs(loc=alpha, scale=sd_a_prop)
        beta_prop = norm.rvs(loc=b, scale=sd_b_prop)

        target_prop = log_conditional_posterior_alpha_beta(alpha_prop, beta_prop, theta_samples[:, i])
        current_prop = log_conditional_posterior_alpha_beta(alpha, b, theta_samples[:, i])

        r_alpha = target_prop - current_prop

        u = np.log(uniform.rvs(loc=0, scale=1))
        if r_alpha >= u:
            alpha = alpha_prop
            b = beta_prop
            num_accepted += 1

        alpha_samples[0, i] = alpha
        beta_samples[0, i] = b

    result = {
        "theta.samp": theta_samples,  # Replace theta_samp with your actual Python variable
        "alpha.samp": alpha_samples,  # Replace alpha_samp with your actual Python variable
        "beta.samp": beta_samples,  # Replace beta_samp with your actual Python variable
        "acceptance_rate": num_accepted / num_samples
    }

    return result


def main():
    samples = 184500
    # samples = 500 + 100*500
    # chain1 = gibbs(.01, .01, 100000, .0525, .12)
    # chain2 = gibbs(3, 3, 100000, .0525, .12)
    # chain3 = gibbs(.2, 1.6, 100000, .0525, .12)
    # chain4 = gibbs(2.46, 0.2, 100000, .0525, .12)

    chain1 = log_gibbs(-2.302586093, -2.302586093, samples, 0.07796, .09221)
    with open('chain1_high.pickle', 'wb') as file:
        pickle.dump(chain1, file)
    print(chain1['acceptance_rate'])

    chain2 = log_gibbs(0.8109302162, 2.302585093, samples, .07796, .09221)
    with open('chain2_high.pickle', 'wb') as file:
        pickle.dump(chain2, file)
    print(chain2['acceptance_rate'])

    chain3 = log_gibbs(-1.3862943611, 1.8245492921, samples, .07796, .09221)
    with open('chain3_high.pickle', 'wb') as file:
        pickle.dump(chain3, file)
    print(chain3['acceptance_rate'])

    chain4 = log_gibbs(0.5596157879, 1.3862943611, samples, .07796, .09221)
    with open('chain4_high.pickle', 'wb') as file:
        pickle.dump(chain4, file)
    print(chain4['acceptance_rate'])

    # reg = np.array([0, 369, 532, 618, 674, 711, 738, 759])

    # holdout_alphas = np.random.choice(final_alpha, size=241, replace=False)
    # holdout_betas = np.random.choice(final_beta, size=241, replace=False)
    #
    # holdout_thetas = [beta.rvs(alpha, b + 7) for alpha, b in zip(holdout_alphas, holdout_betas)]
    #
    # print(holdout_thetas)
    # print(len(holdout_thetas))
    #
    # holdout_samples = [np.random.geometric(theta, 1) for theta in holdout_thetas]
    #
    # # holdout_samples_1d = np.ravel(holdout_samples)
    #
    # y_predictions = []
    # for alpha, b in zip(final_alpha, final_beta):
    #     theta = beta.rvs(alpha, b)  # Sample theta from Beta distribution
    #     y_pred = geom.rvs(theta)  # Sample Y from Geometric distribution
    #     y_predictions.append(y_pred)
    #
    # df = pd.DataFrame(y_predictions, columns=['SampleValue'])
    #
    # # Group by the sample values and count the occurrences
    # counts = df.groupby('SampleValue').size() / len(final_beta) * 1000
    #
    # print(counts[:12])
    #
    # survival = [1000]
    # for i, j in enumerate(counts[:12]):
    #     survival.append(survival[i] - j)
    #
    # print(survival)

    #
    # y_predictions = []
    #
    # for i in range(0, 12): # Sample theta from Beta distribution
    #     y_pred = beta_function(mode_estimate_a + 1, mode_estimate_b + i)/beta_function(mode_estimate_a, mode_estimate_b)
    #     y_predictions.append(y_pred)
    #
    # y_predictions = 1000 * np.array(y_predictions)
    #
    # print(y_predictions[:12])
    #
    # survival = [1000]
    # for i, j in enumerate(y_predictions):
    #     survival.append(survival[i] - j)
    #
    # print(survival)
    # df = pd.DataFrame(y_predictions, columns=['SampleValue'])
    #
    # # Group by the sample values and count the occurrences
    # counts = df.groupby('SampleValue').size()
    #
    # print(counts[:12])

    # mean_a = np.mean(final_alpha)
    # mean_b = np.mean(final_beta)
    #
    # probabilities = np.zeros(20)
    #
    # for i in range(0, 20):
    #     probabilities[i] = beta_function(mean_a + 1, mean_b + 7 + i)/beta_function(mean_a, mean_b + 7)
    #
    # print(241*probabilities)

    # print(holdout_samples_1d)
    # print(len(holdout_samples_1d))

    #
    #
    # # Now create the DataFrame
    # df = pd.DataFrame(holdout_samples_1d, columns=['SampleValue'])
    #
    # # Group by the sample values and count the occurrences
    # counts = df.groupby('SampleValue').size()
    #
    # print(counts)
    #
    # probabilities = np.zeros(20)
    #
    # for i in range(0, 20):
    #     probabilities[i] = [beta_function(alpha + 1, b + i)/beta_function(alpha, b) for alpha, b in zip(holdout_alphas, holdout_betas)]
    #
    # counts = 241 * probabilities
    #
    # # print(counts)
    # ess = calculate_ess(final_alpha)
    # print("Effective Sample Size:", ess)

    # print(len(final_alpha) / 100)


if __name__ == '__main__':
    main()
