import numpy as np
import matplotlib.pyplot as plt
from Algorithms.HMC import *
from Algorithms.Other_MCMC_algo import *
from scipy.optimize import minimize
import seaborn as sns

np.random.seed(123)

# Experiment 2: ring shape distribution

def u_shaped_distribution(x, y):
    return 0.5 * (x**2 + y**2) - 0.5 * (x**2 - y**2)**2

def log_u_shaped_distribution(x, y):
    return -0.5 * (x**2 + y**2) - 0.5 * (x**2 - y**2)**2

def grad_log_u_shaped_distribution(x, y):
    return np.array([-x - 2 * x * (x**2 - y**2), -y - 2 * y * (x**2 - y**2)])

logp = lambda x: log_u_shaped_distribution(x[0], x[1])
U = lambda x: u_shaped_distribution(x[0], x[1])
gradU = lambda x: grad_log_u_shaped_distribution(x[0], x[1])

T = 2000
burnin = 1000

theta0 = np.array([0, 0])

thetas_MH = Metropolis_Hastings(logp, theta0, scale=1, T=T, burnin=burnin, thin=1)
thetas_Gibbs = Gibbs_within_MH_rand(logp, theta0, scale=1, T=T, burnin=burnin, thin=1)
thetas_HMC = HMC(U, gradU, theta0, L=10, eps=0.1, scale=1, T=T, burnin=burnin, thin=1)

# Plot the trajectory of the Markov Chain for the three algorithms in a contour plot of the distribution
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = u_shaped_distribution(X, Y)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].contour(X, Y, Z)
ax[0].plot(thetas_HMC[:, 0], thetas_HMC[:, 1], color='r')
ax[0].set_title('HMC')
ax[1].contour(X, Y, Z)
ax[1].plot(thetas_MH[:, 0], thetas_MH[:, 1], color='r')
ax[1].set_title('RH')
ax[2].contour(X, Y, Z)
ax[2].plot(thetas_Gibbs[:, 0], thetas_Gibbs[:, 1], color='r')
ax[2].set_title('RHWG')
plt.tight_layout()
plt.savefig('contour.png')
plt.show()