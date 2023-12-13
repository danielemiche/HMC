import numpy as np
import scipy.stats as sps
from tqdm import tqdm

def HMC_acceptance(U, gradU, eps, L, current_x, scale=1):
    current_p = sps.multivariate_normal(np.zeros(len(current_x)), scale*np.eye(len(current_x))).rvs(1)
    p = current_p.copy()
    x = current_x.copy()
    H = U(x) + 0.5 * np.dot(p, p)/scale
    p -= eps/2 * gradU(x)
    for l in range(L):
        x += eps * p
        if l!=L-1:
            p -= eps * gradU(x)
    p -= eps/2 * gradU(x)
    p = -p
    H_new = U(x) + 0.5 * np.dot(p, p)/scale
    if np.log(sps.uniform().rvs(1))<H-H_new:
        return x
    else:
        return current_x
    
def HMC(U, gradU, theta0, L, eps, scale=1, T=10000, burnin=1000, thin=1):
    nparams = len(theta0)
    thetas = np.zeros((T+burnin, nparams))
    thetas[0] = theta0
    for _, i in enumerate(tqdm(range(1, T+burnin))):
        thetas[i] = HMC_acceptance(U, gradU, eps, L, thetas[i-1], scale=scale)
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

