import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from leapfrog import H, leapfrog

@jit(nopython=True)
def metropolis_hastings_step(H, phi0, pars):
    p0 = np.random.normal(0,1)
    p, phi = leapfrog(p0, phi0, Nmd, pars)
    if np.random.uniform(0,1) <= np.exp(H(p0,phi0,pars)-H(p,phi,pars)):
        return True, phi
    else:
        return False, phi0

def bootstrap_error(array, nBS):
    # performs a bootstrap error estimation for an array, generating nBS bootstrap samples  
    n = len(array)
    bsmean = np.zeros(nBS)
    for i in range(nBS):
        indices = np.random.randint(n,size=n) # random bootstrap indices
        bsmean[i] = array[indices].mean()
    return bsmean.std()

#@jit(nopython=True)
def simulate(Ncfg, Ntherm, Nmd, nBS, pars):
    N, beta, J, h = pars
    
    # thermalize
    phi0 = 0.0
    for i in range(Ntherm):
        _, phi0 = metropolis_hastings_step(H, phi0, pars)
    
    phi_array = np.zeros(Ncfg)
    counter = 0
    for i in range(Ncfg):
        accepted, phi_array[i] = metropolis_hastings_step(H, phi0, pars)
        if accepted:
            counter += 1
    
    counter /= Ncfg
    
    m = np.tanh(beta*h+phi_array)
    dm = bootstrap_error(m, nBS)
    m = np.mean(m)
    
    eps = 1/(2*beta*N) - phi_array**2/(2*J*beta**2) - h*np.tanh(beta*h+phi_array)
    deps = bootstrap_error(eps, nBS)
    eps = np.mean(eps)
    
    return m, dm, eps, deps, counter
    

N = 10.
beta = 1.
J = None
h = 0.5
pars = [N, beta, J, h]

Ncfg = 100
Ntherm = 1000
Nmd = 100
nBS = 500

m = np.zeros(20)
dm = m.copy()
eps = m.copy()
deps = m.copy()
acceptance = m.copy()
J_array = np.linspace(0.2,2,20)
for i, J in enumerate(J_array):
    pars[2] = J
    
    m[i], dm[i], eps[i], deps[i], acceptance[i] = simulate(Ncfg, Ntherm, Nmd, nBS, pars)

plt.errorbar(J_array,m,dm)
plt.scatter(J_array,acceptance)
plt.show()
