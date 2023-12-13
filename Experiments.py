import numpy as np
import matplotlib.pyplot as plt
from HMC import *
from Other_MCMC_algo import *
import scipy.stats as sps
import scipy.special as spsp
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import seaborn as sns

# Bayesian Logistic Regression 

# def log_prior(theta):
#     return -0.5 * np.sum(theta**2)
s = 100
def log_prior(theta):
    return sps.norm(0, s).logpdf(theta).sum()
def dlog_prior(theta):
    return -theta/s**2


def log_likelihood(theta, X, y):
    return np.sum(np.log(1/(1 + np.exp(-X @ theta))) * y +(1 - y) * np.log(1/(1 + np.exp(X @ theta))))   
def dlog_likelihood(theta, X, y):
    return X.T @ (y - 1/(1 + np.exp(-X @ theta)))

def log_posterior(theta, X, y):
    return log_prior(theta) + log_likelihood(theta, X, y)
def dlog_posterior(theta, X, y):
    return dlog_prior(theta) + dlog_likelihood(theta, X, y)

betas = np.array([.1, -.2, 1])
num_dims = len(betas)
N = 500

X = np.random.randn(N, num_dims)
p = 1/(1 + np.exp(-X @ betas))
y = sps.bernoulli(p).rvs(N)

MLE = minimize(lambda theta: -log_likelihood(theta, X, y), np.zeros(num_dims)).x
MAP = minimize(lambda theta: -log_posterior(theta, X, y), np.zeros(num_dims)).x


betas_HMC = HMC(log_posterior, dlog_posterior, np.zeros(num_dims), L=10, eps=0.1, scale=1, T=10000, burnin=1000, thin=1)

