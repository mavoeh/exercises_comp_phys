import numpy as np
from numba import jit

@jit(nopython=True)
def H(p, phi, pars):
    # returns the value of the hamiltonian for the parameters in pars = (N, beta, h, J)
    N, beta, J, h = pars
    return p**2/2 + phi**2*N/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))

@jit(nopython=True)
def leapfrog(p, phi, Nmd, pars):
    # implements leapfrog for this problem using parameters pars = (N, beta, h, J)
    # returns p_f and phi_f
    N, beta, J, h = pars
    eps = 1/Nmd
    phi += eps*p/2.
    for i in range(Nmd-1):
        p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
        phi += eps*p
    p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
    phi += eps*p/2
    return p, phi
