import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def H(p, phi, pars):
    N, beta, J, h = pars
    return p**2/2 + phi**2*N/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))

@jit(nopython=True)
def leapfrog(p, phi, Nmd, pars):
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
N = 10
beta = 1
J = 0.5
h = 0.5
pars = (N, beta, J, h)

Nmd_max = 100
p0, phi0 = 0, 0
H0 = H(p0,phi0, pars)
y = np.zeros(Nmd_max)
for i, Nmd in enumerate(range(1,Nmd_max+1)):
    p, phi = leapfrog(p0, phi0, Nmd, pars)
    y[i] = H(p,phi,pars)
y = (y-H0)/H0

plt.scatter(range(1,Nmd_max+1),y)
plt.yscale("log")
plt.ylim(1e-3,1e-1)
plt.show()

