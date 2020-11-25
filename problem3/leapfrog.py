import numpy as np
import matplotlib.pyplot as plt
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
    phi += eps/2
    for i in range(Nmd-1):
        p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
        phi += eps*p
    p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
    phi += eps*p/2
    return p, phi


# test leapfrog
def test_leapfrog(Nmd_max,pars):
    fig = plt.figure()
    N, beta, J, h = pars
    # test for random values of p, phi
    p0, phi0 = np.random.normal(0,1),np.random.normal(-1,1)
    # calculate Hamiltonian for starting values of p, phi
    H0 = H(p0,phi0,pars)
    y = np.zeros(Nmd_max) # array to save difference of hamiltonians for all Nmds
    for i, Nmd in enumerate(range(1,Nmd_max+1)):
        p, phi = leapfrog(p0, phi0, Nmd, pars)
        y[i] = H(p,phi,pars) # store Hamiltonian for pf, phif
    y = np.abs(y-H0)/np.abs(H0) # calculate relative difference
    
    # plot
    plt.scatter(range(1,Nmd_max+1),y)
    plt.yscale("log")
    # set axis range
    plt.ylim(min(y)-0.1*10**np.log10(min(y)),max(y[5:])+0.5*10**np.log10(max(y[5:])))
    plt.xlim(0,Nmd+1)
    plt.xlabel("$N_\mathrm{md}$", fontsize = 18)
    plt.ylabel(r"$\frac{|H(p_f,\phi_f)-H(p_0,\phi_0)|}{|H(p_0,\phi_0)|}$", fontsize = 22)
    plt.title(r"$N={0:.0f}\quad\beta={1:.1f}\quad h={2:.1f}\quad J={3:.1f}\quad p_0={4:.2f}\quad\phi_0={5:.2f}$".format(*pars,p0,phi0),fontsize=18) 
    plt.grid(True)
    plt.tight_layout()
    return fig

#test_leapfrog(200,(10,1,0.5,0.5)).savefig("test_leapfrog.pdf")
