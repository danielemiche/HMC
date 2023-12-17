import numpy as np
import scipy.stats as sps
from tqdm import tqdm


def Metropolis_Hastings(logp, theta0, scale = 1, T=10000, burnin=1000, thin=1):
    nparams = len(theta0)
    thetas = np.zeros((T+burnin, nparams))
    thetas[0] = theta0
    logp0 = logp(theta0)
    for i, j in enumerate(tqdm(range(1, T+burnin))):
        z = sps.norm(thetas[i-1], scale=scale).rvs(nparams)
        logpz = logp(z)
        u = sps.uniform().rvs(1)
        if np.log(u)<logpz-logp0:
            thetas[i] = z
            logp0 = logpz
        else:
            thetas[i] = thetas[i-1]            
    return thetas[::thin, :]

def Gibbs_within_MH_rand(logp, theta0, scale = 1, T=10000, burnin=1000, thin=1):
    nparams = len(theta0)
    thetas = np.zeros((T+burnin, nparams))
    thetas[0] = theta0
    logp0 = logp(theta0)
    for i, j in enumerate(tqdm(range(1, T+burnin))):
        k = np.random.randint(nparams)
        z = thetas[i-1].copy()
        z[k] = sps.norm(thetas[i-1][k], scale=scale).rvs(1)
        logpz = logp(z)
        u = sps.uniform().rvs(1)
        if np.log(u)<logpz-logp0:
            thetas[i] = z
            logp0 = logpz
        else:
            thetas[i] = thetas[i-1]            
    return thetas[::thin, :]

def ULA(gradU, theta0, gamma=0.1, T=10000, burnin=1000, thin=1):
    nparams = len(theta0)
    thetas = np.zeros((T+burnin, nparams))
    thetas[0] = theta0
    for _, i in enumerate(tqdm(range(1, T+burnin))):
        z = sps.multivariate_normal(np.zeros(nparams), np.eye(nparams)).rvs(1)
        thetas[i] = thetas[i-1] - gamma * gradU(thetas[i-1]) + np.sqrt(2*gamma)*z
    thetas = thetas[burnin:, :]
    return thetas[::thin, :]

def MALA(logp, gradU, theta0, gamma=0.1, T=10000, burnin=1000, thin=1):
    nparams = len(theta0)
    thetas = np.zeros((T+burnin, nparams))
    thetas[0] = theta0
    grad0 = gradU(theta0)
    logp0 = logp(theta0)
    for _, i in enumerate(tqdm(range(1, T+burnin))):
        mu = thetas[i-1] - gamma * grad0
        sigma = np.sqrt(2*gamma)
        z = sps.multivariate_normal(mu, np.eye(nparams)*sigma**2).rvs(1)
        u = sps.uniform().rvs(1)
        gradz = gradU(z)
        logpz = logp(z)
        qz0 = sps.multivariate_normal.logpdf(thetas[i-1], z-gamma*gradz, sigma**2*np.eye(nparams))
        q0z = sps.multivariate_normal.logpdf(z, mu, 2*gamma*np.eye(nparams))
        if np.log(u) < logpz - logp0 + qz0 - q0z:
            thetas[i] = z
            grad0 = gradz
            logp0 = logpz
        else:
            thetas[i] = thetas[i-1]
    thetas = thetas[burnin:, :]
    return thetas[::thin, :]

# Utils 
def find_MAP(sample):
    if len(sample.shape)==1:
        h = np.histogram(sample, bins = 50)
        MAP = h[1][np.argmax(h[0])]
    else:
        MAP = np.zeros(sample.shape[1])
        for i in range(sample.shape[1]):
            h = np.histogram(sample[:, i], bins = 50)
            MAP[i] = h[1][np.argmax(h[0])]
    return MAP
