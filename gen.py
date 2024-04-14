import distutils.util
import pickle
import sys
import textwrap

import arviz as az
import matplotlib
import numpy as np
import torch
from scipy.special import beta as beta_func
from torch.distributions import Beta, geometric

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import MCMC
from gen_multinomial_metric import sBG


def survival(alpha, beta, t):
    return beta_func(alpha, (beta + t)) / beta_func(alpha, beta)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python gen.py <number_of_samples> <number_of_chains> <sample_boolean>"
        )
        return
    int_arg = int(sys.argv[1])
    chains_arg = int(sys.argv[2])
    bool_arg = bool(distutils.util.strtobool(sys.argv[3]))

    num_samples = int_arg
    num_adapt = 2000
    end = num_samples - num_adapt
    num_chains = chains_arg
    sample = bool_arg

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    data = np.array([131, 126, 90, 60, 42, 34, 26, 491])

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 18, 16, 13, 194])
    # data = np.array([131, 126, 90, 60, 42, 34, 26, 23, 23, 18, 427])

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 18, 16, 13, 11, 10, 173])
    # data = np.array([152, 103, 75, 58, 46, 38, 32, 27, 24, 21, 18, 16, 15, 13, 12, 11, 10, 9, 318])

    if sample:
        chains = sBG(data, num_samples, num_adapt, num_chains)
    else:
        chains = []
        for i in range(1, num_chains + 1):
            file_path = f"NUTS/chain{i}_high_HMC.pickle"
            with open(file_path, "rb") as file:
                chain = pickle.load(file)
                chains.append(chain)

    samples = []
    samples_final = []
    for chain in chains:
        sample = {"alpha": chain["alpha.samp"][0, :], "beta": chain["beta.samp"][0, :]}
        sample_final = {
            "alpha": chain["alpha.samp"][0, num_adapt:num_samples],
            "beta": chain["beta.samp"][0, num_adapt:num_samples],
        }
        samples.append(sample)
        samples_final.append(sample_final)

    # Warm-up removal and concatenation of parameter chains
    final_params = {"alpha": np.array([]), "beta": np.array([])}

    for sample in samples:
        final_params["alpha"] = np.concatenate(
            (final_params["alpha"], sample["alpha"][num_adapt:num_samples])
        )
        final_params["beta"] = np.concatenate(
            (final_params["beta"], sample["beta"][num_adapt:num_samples])
        )

    print(len(final_params["alpha"]))

    alphas = final_params["alpha"]
    betas = final_params["beta"]

    # Plots an animation of chain sampling with and without warm-up samples
    MCMC.animate_chains(
        samples=samples,
        num_frames=2000,
        interval=20,
        x_lim=(0, 4),
        y_lim=(0, 4),
    )

    MCMC.animate_chains(
        samples=samples_final,
        num_frames=2000,
        interval=20,
    )

    # Plots a set of chains for the alpha parameters
    MCMC.plot_chains(
        samples,
        parameter_name="alpha",
        begin=0,
        end=1000,
        y_lim=(0, 3),
    )
    MCMC.plot_chains(
        samples_final,
        parameter_name="alpha",
        begin=0,
        end=end,
        y_lim=(0, 2),
    )

    # Plots a set of chains for the beta parameters
    MCMC.plot_chains(
        samples,
        parameter_name="beta",
        begin=0,
        end=1000,
        y_lim=(0, 4),
    )
    MCMC.plot_chains(
        samples_final,
        parameter_name="beta",
        colors=None,
        begin=0,
        end=end,
        y_lim=(0, 10),
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plots the posterior beliefs of alpha and beta
    az.plot_posterior(final_params["alpha"], ref_val=0.668, ax=axs[0])
    axs[0].set_title("Posterior Belief in Alpha (Action Parameter)", fontsize=18)
    axs[0].set_xlabel("Value of Alpha", fontsize=14)
    axs[0].set_ylabel("Density", fontsize=14)
    axs[0].set_xlim([0.5, 8.5])

    az.plot_posterior(final_params["beta"], ref_val=3.802, ax=axs[1])
    axs[1].set_title("Posterior Belief in Beta (Inaction Parameter)", fontsize=18)
    axs[1].set_xlabel("Value of Beta", fontsize=14)
    axs[1].set_ylabel("Density", fontsize=14)
    axs[1].set_xlim([2, 10])

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 1, squeeze=True)
    plot_acf(final_params["alpha"], lags=100, ax=ax[0])
    plot_acf(final_params["beta"], lags=999, ax=ax[1])
    plt.show()

    # Plots posterior beliefs of a customer's propensity to churn given their observed beahavior.
    true_index = 0
    num_rows = (len(data) + 3) // 4  # Calculate number of rows needed

    fig, axs = plt.subplots(
        num_rows,
        4,
        figsize=(15, 3 * num_rows),
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    for i, val in enumerate(data):
        final_theta_i = np.concatenate(
            [chains[j]["theta.samp"][true_index, num_adapt:] for j in range(num_chains)]
        )

        y_max = 7
        ax = axs[i // 4, i % 4]  # Get the subplot for this graph
        ax.hist(final_theta_i, bins=30, density=True)
        # az.plot_posterior(final_theta_i, kind="kde", ax=ax)

        title_text = (
            "Posterior Belief for a Customer's Theta Who Churned at Period " + str(i + 1)
        ) * ((i + 1) % len(data) != 0) + (
            "Postersior Belief for a Customer's Theta Who Survived Through Period " + str(i)
        ) * (
            (i + 1) % len(data) == 0
        )
        wrapped_text = textwrap.fill(title_text, width=40)  # Adjust the width as needed

        ax.set_title(wrapped_text)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Theta Value")
        ax.set_ylabel("Density")
        true_index += val

    plt.suptitle(
        "Posterior Beliefs of Churn Propensity for Regular Customers of Similar Behavior",
        fontsize=22,
    )
    plt.tight_layout()
    plt.show()

    num_samples = len(samples_final[0]["alpha"])

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

    print(az.rhat(inference_data))
    print(az.hdi(inference_data, hdi_prob=0.95))

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 18, 16, 13, 11, 10, 173])
    data = np.array([131, 126, 90, 60, 42, 34, 26, 23, 23, 18, 18, 15, 394])
    alive = np.concatenate(([1000], (np.sum(data) - np.cumsum(data))))
    x1 = range(0, len(alive) - 1)

    survivals = np.zeros((16, len(alphas)))

    weeks = np.arange(0, 16)
    for i in range(len(alphas)):  # Iterate over alpha-beta pairs
        for j, week in enumerate(weeks):  # Iterate over weeks
            survivals[j, i] = survival(alphas[i], betas[i], week + 1)

    eb_alpha = 0.668
    eb_beta = 3.802
    eb_survival = np.zeros(16)

    for i, t in enumerate(weeks):
        eb_survival[i] = 1000 * survival(eb_alpha, eb_beta, t + 1)

    survivals = survivals * 1000
    weeks = np.arange(0, 17)
    initial = np.full((1, len(alphas)), 1000)
    survivals = np.concatenate((initial, survivals))
    eb_survival = np.concatenate(([1000], eb_survival))
    hdi_data = np.array([az.hdi(week_data, hdi_prob=0.95) for week_data in survivals])

    az.plot_hdi(weeks, hdi_data=hdi_data, hdi_prob=0.95, color="C0")
    plt.axvspan(None, None, color="C0", alpha=0.7, label="95% HDI")
    plt.plot(x1, alive[:-1], label="Observed", color="black")
    plt.plot(weeks, eb_survival, label="Empirical Bayes", color="red")
    plt.axvline(7, ls="--", color="k")
    plt.legend()
    plt.title("Customer Survival Curves", fontsize=18)
    plt.xlabel("Customer Tenure (Years)", fontsize=14)
    plt.ylabel("Customers Alive", fontsize=14)
    plt.ylim(0, 1000)
    plt.show()


if __name__ == "__main__":
    main()
