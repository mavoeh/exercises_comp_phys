import numpy as np
import matplotlib.pyplot as plt

N = 100
beta = 1
h = 0.5
J = 1

def H(p, phi):
    return p**2/2 + phi**2/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))

def leapfrog(func, p, phi, Nmd):
    eps = 1/Nmd
    phi += eps/2 * p
    for n in range(Nmd-1):
        p -= eps*(phi/(beta*J) - N*np.tanh(beta*h+phi))
        phi += eps*p
    p -= eps*(phi/(beta*J) - N*np.tanh(beta*h+phi))
    phi += eps/2*p
    return p, phi

Nmd_max = 100
p0 = 1
phi0 = 1
H0 = H(p0, phi0)
x = range(1,Nmd_max+1)
y = np.zeros(Nmd_max)
for Nmd in x:
    y[Nmd-1] = np.abs(H(*leapfrog(H,p0,phi0,Nmd)) - H0)
y /= np.abs(H0)

print(y)
plt.scatter(x,y)
plt.xlim(0,Nmd+1)
plt.yscale("log")
plt.ylim(2e-4,1e1)
plt.grid(True)
plt.xlabel(r"$N_\mathrm{md}$",fontsize=18)
plt.ylabel(r"$\frac{|H(p_f,\phi_f)-H(p_0,\phi_0)|}{|H(p_0,\phi_0)|}$",fontsize=20)

plt.show()
    
