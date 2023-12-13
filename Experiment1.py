import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
import scipy.stats as sps
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
import seaborn as sns

np.random.seed(123)
# Experiment 1: Bayesian Logistic Regression 

s = 10
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

logp = lambda theta: log_posterior(theta, X, y)
U = lambda theta: -logp(theta)
gradU = lambda theta: -dlog_posterior(theta, X, y)

MLE = minimize(lambda theta: -log_likelihood(theta, X, y), np.zeros(num_dims)).x

T = 10000
burnin = 1000

betas_MH = Metropolis_Hastings(logp, np.zeros(num_dims), scale=1, T=T, burnin=burnin, thin=1)
betas_Gibbs = Gibbs_within_MH_rand(logp, np.zeros(num_dims), scale=1, T=T, burnin=burnin, thin=1)
betas_HMC = HMC(U, gradU, np.zeros(num_dims), L=10, eps=0.1, scale=1, T=T, burnin=burnin, thin=1)

# print a nice table containing all the estimates for the betas
table = np.zeros((num_dims, 5))
table[:, 0] = betas
table[:, 1] = MLE
table[:, 2] = np.mean(betas_HMC, axis=0)
table[:, 3] = np.mean(betas_MH, axis=0)
table[:, 4] = np.mean(betas_Gibbs, axis=0)

table = pd.DataFrame(table, columns=['True', 'MLE', 'HMC', 'RH', 'RHWG'])
table.index = ['$beta_{}$'.format(i+1) for i in range(num_dims)]
latex_table = table.to_latex(index=True)
print(latex_table)

# Plot the histograms of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))

sns.histplot(betas_HMC[:, 0], bins=50, kde=True, ax=ax[0, 0])
sns.histplot(betas_HMC[:, 1], bins=50, kde=True, ax=ax[1, 0])
sns.histplot(betas_HMC[:, 2], bins=50, kde=True, ax=ax[2, 0])
sns.histplot(betas_MH[:, 0], bins=50, kde=True, ax=ax[0, 1])
sns.histplot(betas_MH[:, 1], bins=50, kde=True, ax=ax[1, 1])
sns.histplot(betas_MH[:, 2], bins=50, kde=True, ax=ax[2, 1])
sns.histplot(betas_Gibbs[:, 0], bins=50, kde=True, ax=ax[0, 2])
sns.histplot(betas_Gibbs[:, 1], bins=50, kde=True, ax=ax[1, 2])
sns.histplot(betas_Gibbs[:, 2], bins=50, kde=True, ax=ax[2, 2])

# Set labels so that there are betas on the y axis only for HMC and make the other not visible
for i in range(3):
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])


# Plot the true value
for i in range(3):
    ax[i, 0].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 1].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 2].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 0].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 1].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 2].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)


ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')

ax[0, 0].legend()
plt.tight_layout()
plt.savefig('hist_log.png')
plt.show()

# Plot the traceplots of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))
for i in range(3):
    ax[i, 0].plot(betas_HMC[:, i])
    ax[i, 1].plot(betas_MH[:, i])
    ax[i, 2].plot(betas_Gibbs[:, i])
    ax[i, 0].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 1].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 2].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 0].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 1].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 2].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])
ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')
ax[0, 0].legend()
plt.tight_layout()
plt.savefig('trace_log.png')
plt.show()

# Plot the autocorrelation of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))
for i in range(3):
    ax[i, 0].acorr(betas_HMC[:, i] - np.mean(betas_HMC[:, i]), maxlags=100)
    ax[i, 1].acorr(betas_MH[:, i] - np.mean(betas_MH[:, i]), maxlags=100)
    ax[i, 2].acorr(betas_Gibbs[:, i] - np.mean(betas_Gibbs[:, i]), maxlags=100)
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])
ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')
plt.tight_layout()
plt.savefig('autocorr_log.png')
plt.show()







=======
import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
import scipy.stats as sps
# import scipy.special as spsp
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
import seaborn as sns

np.random.seed(123)
# Bayesian Logistic Regression 

# def log_prior(theta):
#     return -0.5 * np.sum(theta**2)
s = 10
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

logp = lambda theta: log_posterior(theta, X, y)
U = lambda theta: -logp(theta)
gradU = lambda theta: -dlog_posterior(theta, X, y)

MLE = minimize(lambda theta: -log_likelihood(theta, X, y), np.zeros(num_dims)).x

T = 10000
burnin = 1000

betas_MH = Metropolis_Hastings(logp, np.zeros(num_dims), scale=1, T=T, burnin=burnin, thin=1)
betas_Gibbs = Gibbs_within_MH_rand(logp, np.zeros(num_dims), scale=1, T=T, burnin=burnin, thin=1)
betas_HMC = HMC(U, gradU, np.zeros(num_dims), L=10, eps=0.1, scale=1, T=T, burnin=burnin, thin=1)

# print a nice table containing all the estimates for the betas
table = np.zeros((num_dims, 5))
table[:, 0] = betas
table[:, 1] = MLE
table[:, 2] = np.mean(betas_HMC, axis=0)
table[:, 3] = np.mean(betas_MH, axis=0)
table[:, 4] = np.mean(betas_Gibbs, axis=0)

table = pd.DataFrame(table, columns=['True', 'MLE', 'HMC', 'RH', 'RHWG'])
table.index = ['$beta_{}$'.format(i+1) for i in range(num_dims)]
latex_table = table.to_latex(index=True)
print(latex_table)

# Plot the histograms of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))

sns.histplot(betas_HMC[:, 0], bins=50, kde=True, ax=ax[0, 0])
sns.histplot(betas_HMC[:, 1], bins=50, kde=True, ax=ax[1, 0])
sns.histplot(betas_HMC[:, 2], bins=50, kde=True, ax=ax[2, 0])
sns.histplot(betas_MH[:, 0], bins=50, kde=True, ax=ax[0, 1])
sns.histplot(betas_MH[:, 1], bins=50, kde=True, ax=ax[1, 1])
sns.histplot(betas_MH[:, 2], bins=50, kde=True, ax=ax[2, 1])
sns.histplot(betas_Gibbs[:, 0], bins=50, kde=True, ax=ax[0, 2])
sns.histplot(betas_Gibbs[:, 1], bins=50, kde=True, ax=ax[1, 2])
sns.histplot(betas_Gibbs[:, 2], bins=50, kde=True, ax=ax[2, 2])

# Set labels so that there are betas on the y axis only for HMC and make the other not visible
for i in range(3):
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])


# Plot the true value
for i in range(3):
    ax[i, 0].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 1].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 2].axvline(x=betas[i], color='r', linestyle='-', label='True')
    ax[i, 0].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 1].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 2].axvline(x=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)


ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')

ax[0, 0].legend()
plt.tight_layout()
plt.savefig('hist_log.png')
plt.show()

# Plot the traceplots of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))
for i in range(3):
    ax[i, 0].plot(betas_HMC[:, i])
    ax[i, 1].plot(betas_MH[:, i])
    ax[i, 2].plot(betas_Gibbs[:, i])
    ax[i, 0].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 1].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 2].axhline(y=betas[i], color='r', linestyle='-', label='True')
    ax[i, 0].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 1].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 2].axhline(y=MLE[i], color='g', linestyle='--', label='MLE', alpha=0.5)
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])
ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')
ax[0, 0].legend()
plt.tight_layout()
plt.savefig('trace_log.png')
plt.show()

# Plot the autocorrelation of the estimates for the beta
fig, ax = plt.subplots(3, 3, figsize=(15, 5))
for i in range(3):
    ax[i, 0].acorr(betas_HMC[:, i] - np.mean(betas_HMC[:, i]), maxlags=100)
    ax[i, 1].acorr(betas_MH[:, i] - np.mean(betas_MH[:, i]), maxlags=100)
    ax[i, 2].acorr(betas_Gibbs[:, i] - np.mean(betas_Gibbs[:, i]), maxlags=100)
    ax[i, 0].set_ylabel('Beta {}'.format(i+1))
    ax[i, 1].set_yticklabels([])
    ax[i, 2].set_yticklabels([])
ax[0, 0].set_title('HMC')
ax[0, 1].set_title('RH')
ax[0, 2].set_title('RHWG')
plt.tight_layout()
plt.savefig('autocorr_log.png')
plt.show()






