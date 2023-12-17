import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
import scipy.stats as sps
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns

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
sns.displot(samples_mh, bins=30)

 
plt.subplot(1, subplots, 2)
samples_ula = ULA(gradU, np.ones(num_dims), gamma=0.1, T=num_samples, burnin=burnin, thin=1)
print("Sample mean:", np.mean(samples_ula, axis=0))
print("Sample covariance:", np.cov(samples_ula.T))
sns.displot(samples_ula, bins=30)

plt.subplot(1, subplots, 3)
samples_mala = MALA(logp, gradU, np.ones(num_dims), gamma=0.1, T=num_samples, burnin=burnin, thin=1)
print("Sample mean:", np.mean(samples_mala, axis=0))
print("Sample covariance:", np.cov(samples_mala.T))
sns.displot(samples_mala, bins=30)
plt.show()