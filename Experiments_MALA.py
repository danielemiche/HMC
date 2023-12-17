import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
import scipy.stats as sps
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns

# np.random.seed(123)

experiment_number = 2

if experiment_number == 1:

    # Toy example 1: Normal distribution 

    def log_norm(theta):
        return -0.5 * np.sum(theta**2)
    def dlog_norm(theta):
        return -np.sum(theta)

    logp = lambda theta: log_norm(theta)
    U = lambda theta: -logp(theta)
    gradU = lambda theta: -dlog_norm(theta)

    num_samples = 5000
    num_dims = 1
    burnin = 1000

    subplots = 3

    plt.figure(figsize=(15, 5))

    plt.subplot(1, subplots, 1)
    samples_mh = Metropolis_Hastings(logp, np.ones(num_dims), scale = 1, T=num_samples, burnin=burnin, thin=1)
    print("Sample mean:", np.mean(samples_mh, axis=0))
    print("Sample covariance:", np.cov(samples_mh.T))
    sns.histplot(samples_mh, bins=30, stat='density', color='red', alpha=0.5, kde=True)

    
    plt.subplot(1, subplots, 2)
    samples_ula = ULA(gradU, np.ones(num_dims), gamma=0.1, T=num_samples, burnin=burnin, thin=1)
    print("Sample mean:", np.mean(samples_ula, axis=0))
    print("Sample covariance:", np.cov(samples_ula.T))
    sns.histplot(samples_ula, bins=30, stat='density', color='green', alpha=0.5, kde=True)

    plt.subplot(1, subplots, 3)
    samples_mala = MALA(logp, gradU, np.ones(num_dims), gamma=0.1, T=num_samples, burnin=burnin, thin=1)
    print("Sample mean:", np.mean(samples_mala, axis=0))
    print("Sample covariance:", np.cov(samples_mala.T))
    sns.histplot(samples_mala, bins=30, stat='density', color='blue', alpha=0.5, kde=True)

    plt.tight_layout()

    plt.show()

elif experiment_number == 2:

    # Toy example 2: normal-normal linear model

    def log_prior(theta):
        return -0.5 * np.sum(theta**2)
    def dlog_prior(theta):
        return -theta

    def log_likelihood(theta, X, y):
        return -0.5 * np.sum((y - X @ theta)**2)
    def dlog_likelihood(theta, X, y):
        return -X.T @ (X @ theta - y)
    
    def log_posterior(theta, X, y):
        return log_prior(theta) + log_likelihood(theta, X, y)
    def dlog_posterior(theta, X, y):
        return dlog_prior(theta) + dlog_likelihood(theta, X, y)

    num_dims = 5
    n = 10

    X = np.random.randn(n, num_dims)
    X = np.hstack((np.ones((n, 1)), X))
    betas = np.array([10, 2, 5, -3, 1, 1])
    y = X @ betas + sps.norm(0, 1).rvs(n)  

    logp = lambda theta: log_posterior(theta, X, y)
    U = lambda theta: -logp(theta)
    gradU = lambda theta: -dlog_posterior(theta, X, y)

    num_samples = 8000
    burnin = 1000

    betas_mh = Metropolis_Hastings(logp, np.zeros(num_dims + 1), scale = 1, T=num_samples, burnin=burnin, thin=1)
    betas_ula = ULA(gradU, np.zeros(num_dims + 1), gamma=0.001, T=num_samples, burnin=burnin, thin=1)
    betas_mala = MALA(logp, gradU, np.zeros(num_dims + 1), gamma=0.001, T=num_samples, burnin=burnin, thin=1)

    OLS = np.linalg.inv(X.T @ X) @ X.T @ y
    Ridge = np.linalg.inv(X.T @ X + np.eye(num_dims + 1)) @ X.T @ y

    # print a nice table containing all the estimates for the betas
    table = np.zeros((num_dims + 1, 6))
    table[:, 0] = betas
    table[:, 1] = OLS
    table[:, 2] = Ridge
    table[:, 3] = np.mean(betas_mh, axis=0)
    table[:, 4] = np.mean(betas_ula, axis=0)
    table[:, 5] = np.mean(betas_mala, axis=0)
    
    table = pd.DataFrame(table, columns=['True', 'OLS', 'Ridge', 'MH', 'ULA', 'MALA'])
    table.index = ['$beta_{}$'.format(i+1) for i in range(num_dims + 1)]
    print(table)

    # Plot the histograms of the estimates for the beta
    fig, ax = plt.subplots(3, 3, figsize=(15, 5))

    for i in range(3):
        sns.histplot(betas_mh[:, i], bins=50, kde=True, ax=ax[i, 0])
        sns.histplot(betas_ula[:, i], bins=50, kde=True, ax=ax[i, 1])
        sns.histplot(betas_mala[:, i], bins=50, kde=True, ax=ax[i, 2])
    
    for i in range(3):
        ax[i, 0].set_ylabel('Beta {}'.format(i+1))
        ax[i, 1].set_yticklabels([])
        ax[i, 2].set_yticklabels([])

    for i in range(3):
        ax[i, 0].axvline(x=betas[i], color='r', linestyle='-', label='True')
        ax[i, 1].axvline(x=betas[i], color='r', linestyle='-', label='True')
        ax[i, 2].axvline(x=betas[i], color='r', linestyle='-', label='True')
        ax[i, 0].axvline(x=OLS[i], color='g', linestyle='--', label='OLS', alpha=0.5)
        ax[i, 1].axvline(x=OLS[i], color='g', linestyle='--', label='OLS', alpha=0.5)
        ax[i, 2].axvline(x=OLS[i], color='g', linestyle='--', label='OLS', alpha=0.5)
        ax[i, 0].axvline(x=Ridge[i], color='m', linestyle='--', label='Ridge', alpha=0.5)
        ax[i, 1].axvline(x=Ridge[i], color='m', linestyle='--', label='Ridge', alpha=0.5)
        ax[i, 2].axvline(x=Ridge[i], color='m', linestyle='--', label='Ridge', alpha=0.5)



    ax[0, 0].set_title('MH')
    ax[0, 1].set_title('ULA')
    ax[0, 2].set_title('MALA')

    ax[0, 0].legend()
    plt.tight_layout()
    plt.savefig('hist_lin_13.png')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(15, 5))

    for i in range(3):
        sns.histplot(betas_mh[:, i+3], bins=50, kde=True, ax=ax[i, 0])
        sns.histplot(betas_ula[:, i+3], bins=50, kde=True, ax=ax[i, 1])
        sns.histplot(betas_mala[:, i+3], bins=50, kde=True, ax=ax[i, 2])

    for i in range(3):
        ax[i, 0].set_ylabel('Beta {}'.format(i+4))
        ax[i, 1].set_yticklabels([])
        ax[i, 2].set_yticklabels([])

    for i in range(3):
        j = i + 3
        ax[i, 0].axvline(x=betas[j], color='r', linestyle='-', label='True')
        ax[i, 1].axvline(x=betas[j], color='r', linestyle='-', label='True')
        ax[i, 2].axvline(x=betas[j], color='r', linestyle='-', label='True')
        ax[i, 0].axvline(x=OLS[j], color='g', linestyle='--', label='MLE', alpha=0.5)
        ax[i, 1].axvline(x=OLS[j], color='g', linestyle='--', label='MLE', alpha=0.5)
        ax[i, 2].axvline(x=OLS[j], color='g', linestyle='--', label='MLE', alpha=0.5)
        ax[i, 0].axvline(x=Ridge[j], color='m', linestyle='--', label='Ridge', alpha=0.5)
        ax[i, 1].axvline(x=Ridge[j], color='m', linestyle='--', label='Ridge', alpha=0.5)
        ax[i, 2].axvline(x=Ridge[j], color='m', linestyle='--', label='Ridge', alpha=0.5)

    ax[0, 0].set_title('MH')
    ax[0, 1].set_title('ULA')
    ax[0, 2].set_title('MALA')

    ax[0, 0].legend()
    plt.tight_layout()
    plt.savefig('hist_lin_36.png')

    plt.show()
