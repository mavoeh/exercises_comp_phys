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

#@jit(nopython=True)
def simulate(Ncfg, Ntherm, Nmd, pars):
    N, beta, J, h = pars
    
    # thermalize
    phi0 = 0.0
    for i in range(Ntherm):
        _, phi0 = metropolis_hastings_step(H, phi0, pars)
    
    phi_array = np.zeros(Ncfg)
    counter = 0
    for i in range(Ncfg):
        accepted, phi = metropolis_hastings_step(H, phi0, pars)
        if accepted:
            counter += 1
    
    counter /= Ncfg
    
    m = np.tanh(beta*h+phi)
    m = m.mean()
    
    eps = 1/(2*beta*N) - phi**2/(2*J*beta**2) - h*np.tanh(beta*h+phi)
    eps = eps.mean()
    
    return m, eps
    

N = 10.
beta = 1.
J = None
h = 0.5
pars = [N, beta, J, h]

Ncfg = 1000
Ntherm = 500
Nmd = 2000

m_array = np.zeros(20)
eps_array = np.zeros(20)
J_array = np.linspace(0.2,2,20)
for i, J in enumerate(J_array):
    pars[2] = J
    
    m_array[i], eps_array[i] = simulate(Ncfg, Ntherm, Nmd, pars)

plt.scatter(J_array,m_array)
plt.show()
