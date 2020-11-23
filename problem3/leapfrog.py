import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit


beta = 1    #beta = 1/kB/T
h = 0.5     #coupling to extenal magnet field

@jit(nopython=True)
def H(p, phi):
    return p**2/2 + phi**2/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))

@jit(nopython=True)
def leapfrog(p, phi, Nmd, N, J):
    eps = 1/Nmd
    phi += eps/2 * p
    for n in range(Nmd-1):
        p -= eps*(phi/(beta*J) - N*np.tanh(beta*h+phi))
        phi += eps*p
    p -= eps*(phi/(beta*J) - N*np.tanh(beta*h+phi))
    phi += eps/2*p
    return p, phi


#Check convergence of leap-frog integration
J = 1       #coupling J
N = 100     #number of spins

Nmd_max = 100
p0 = 1
phi0 = 1
H0 = H(p0, phi0)
x = range(1,Nmd_max+1)
y = np.zeros(Nmd_max)
for Nmd in x:
    y[Nmd-1] = np.abs(H(*leapfrog(p0,phi0,Nmd,N,J)) - H0)
y /= np.abs(H0)

fig_leapfrog = plt.figure()
plt.scatter(x,y)
plt.xlim(0,Nmd+1)
plt.yscale("log")
plt.ylim(2e-4,1e1)
plt.grid(True)
plt.xlabel(r"$N_\mathrm{md}$",fontsize=18)
plt.ylabel(r"$\frac{|H(p_f,\phi_f)-H(p_0,\phi_0)|}{|H(p_0,\phi_0)|}$",fontsize=20)

fig_leapfrog.savefig("leapfrog.pdf")


n_p, n_phi = (101, 401)
p_array = np.linspace(0,1,n_p)
phi_array = np.linspace(-10,10,n_phi)
Nmd = 50

p, phi = np.meshgrid(p_array, phi_array)
H_array = np.zeros((n_phi,n_p))

for i in range(n_phi):
    for j in range(n_p):
        pars = (p[i,j], phi[i,j])
        H_array[i,j] = H(*leapfrog(*pars, Nmd, 10, 0.5)) - H(*pars)


plt.figure()
extent = (p_array.min(),p_array.max(),phi_array.min(),phi_array.max())
plt.imshow(H_array,aspect='auto',origin='lower',extent=extent)
plt.xlabel("$p_0$",fontsize=18)
plt.ylabel("$\phi_0$",fontsize=18)
plt.title("$H(p_f,\phi_f)-H(p_0,\phi_0)$",fontsize=20)
plt.colorbar()
plt.show()
