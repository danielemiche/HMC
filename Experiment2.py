import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
from scipy.optimize import minimize
import seaborn as sns

# Toy example 2: 2D distributions

def log_u_shaped_distr(theta1, theta2):
    return -theta1**2 + theta2**2 - 0.5 * theta2**4

def log_u_shaped_distr_grad(theta1, theta2):
    return np.array([-2 * theta1, 2 * theta2 - 2 * theta2**3])

logp = lambda theta: log_u_shaped_distr(theta[0], theta[1])
U = lambda theta: -logp(theta)
gradU = lambda theta: -log_u_shaped_distr_grad(theta[0], theta[1])

T = 2000
burnin = 1000

thetas_MH = Metropolis_Hastings(logp, np.zeros(2), scale=1, T=T, burnin=burnin, thin=1)
thetas_Gibbs = Gibbs_within_MH_rand(logp, np.zeros(2), scale=1, T=T, burnin=burnin, thin=1)
thetas_HMC = HMC(U, gradU, np.ones(2), L=20, eps=0.01, scale=1, T=T, burnin=burnin, thin=1)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(log_u_shaped_distr(X, Y))

subplots = 3

plt.figure(figsize=(15, 5))
plt.subplot(1, subplots, 1)


plt.contour(X, Y, Z)
plt.scatter(thetas_MH[:, 0], thetas_MH[:, 1], s=1, color='red', alpha=0.5, label='MH')
plt.legend()
plt.subplot(1, subplots, 2)
plt.contour(X, Y, Z)
plt.scatter(thetas_Gibbs[:, 0], thetas_Gibbs[:, 1], s=1, color='green', alpha=0.5, label='Gibbs')
plt.legend()
plt.subplot(1, subplots, 3)
plt.contour(X, Y, Z)
plt.scatter(thetas_HMC[:, 0], thetas_HMC[:, 1], s=1, color='blue', alpha=0.5, label='HMC')
plt.legend()
plt.tight_layout()
plt.show()


# plt.contour(X, Y, Z)
# plt.scatter(thetas_MH[:, 0], thetas_MH[:, 1], s=1, color='red', alpha=0.5, label='MH')
# plt.legend()
# plt.show()

# plt.contour(X, Y, Z)

# plt.scatter(thetas_Gibbs[:, 0], thetas_Gibbs[:, 1], s=1, color='green', alpha=0.5, label='Gibbs')
# plt.show()

# plt.contour(X, Y, Z)
# plt.scatter(thetas_HMC[:, 0], thetas_HMC[:, 1], s=1, color='blue', alpha=0.5, label='HMC')
# plt.show()






