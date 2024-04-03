import numpy as np
import pickle
import arviz as az
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from gen_multinomial_metric import sBG
import MCMC


def main():
    num_samples = 11000
    num_adapt = 1000
    end = num_samples - num_adapt
    chains = 4
    sample = False

    # data = np.array([369, 163, 86, 56, 37, 27, 21, 241])
    data = np.array([131, 126, 90, 60, 42, 34, 26, 491])
    if sample :
        chains = sBG(data, num_samples, num_adapt, chains)
    else:
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
            'alpha': chain['alpha.samp'][0, num_adapt:num_samples],
            'beta': chain['beta.samp'][0, num_adapt:num_samples]
        }
        samples.append(sample)
        samples_final.append(sample_final)
    # thin = int((num_samples - warm_up) * 4 / 1000)

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
        final_params['alpha'] = np.concatenate((final_params['alpha'], sample['alpha'][num_adapt:num_samples]))
        final_params['beta'] = np.concatenate((final_params['beta'], sample['beta'][num_adapt:num_samples]))

    print(len(final_params['alpha']))

    az.plot_posterior(final_params['alpha'], ref_val=0.704)
    plt.show()

    az.plot_posterior(final_params['beta'], ref_val=3.86)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chains[0]['alpha.samp'][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chains[1]['alpha.samp'][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chains[2]['alpha.samp'][0, num_adapt:], lags=120, ax=ax)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(chains[3]['alpha.samp'][0, num_adapt:], lags=120, ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(final_params['beta'], lags=999, ax=ax)
    plt.show()
    # Plots posterior beliefs of a customer's propensity to churn given what we observed them doing
    true_index = 0
    for i, val in enumerate(data):
        final_theta_i = np.concatenate((chains[0]['theta.samp'][true_index, num_adapt:], chains[1]['theta.samp'][true_index, num_adapt:], chains[2]['theta.samp'][true_index, num_adapt:], chains[3]['theta.samp'][true_index, num_adapt:]))
        plt.hist(final_theta_i, bins=30, density=True)
        plt.ylim(0, 12)
        # az.plot_posterior(final_theta_i)
        plt.title(("Posterior of Theta for a customer who churns at period " + str(i + 1)) * ((i + 1) % 8 != 0) + (
            "Theta posterior for a customer who survives through period 7") * ((i + 1) % 8 == 0))
        plt.xlabel("Theta Value")
        plt.ylabel("Density")
        plt.show()
        true_index += val

    num_chains = len(samples_final)
    num_samples = len(samples_final[0]['alpha'])

    data_dict = {
    'alpha': np.array([chain['alpha'] for chain in samples_final]),
    'beta': np.array([chain['beta'] for chain in samples_final])
    }

    inference_data = az.from_dict(data_dict, coords={'chain': np.arange(num_chains), 'draw': np.arange(num_samples)},
                                  dims={'alpha': ['chain', 'draw'], 'beta': ['chain', 'draw']})

    ess_results = az.ess(inference_data)
    print('this is just standard ess: ', ess_results)

if __name__ == "__main__":
    main()
