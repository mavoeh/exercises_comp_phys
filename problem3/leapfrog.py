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
def leapfrog(func, p, phi, Nmd, N, J):  #do not use func?
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
    y[Nmd-1] = np.abs(H(*leapfrog(H,p0,phi0,Nmd,N,J)) - H0)
y /= np.abs(H0)

#print(y)
plt.scatter(x,y)
plt.xlim(0,Nmd+1)
plt.yscale("log")
plt.ylim(2e-4,1e1)
plt.grid(True)
plt.xlabel(r"$N_\mathrm{md}$",fontsize=18)
plt.ylabel(r"$\frac{|H(p_f,\phi_f)-H(p_0,\phi_0)|}{|H(p_0,\phi_0)|}$",fontsize=20)

plt.show()
    

#Apply Leap-frog to sample phi in order to calculate average magnetization m and enegy eps
def bootstrap_error(array, nBS):
    # performs a bootstrap error estimation for an array, generating nBS bootstrap samples  
    n = len(array)
    bsmean = np.zeros(nBS)
    for i in range(nBS):
        indices = np.random.randint(n,size=n) # random bootstrap indices
        bsmean[i] = array[indices].mean()
    return bsmean.std()


@jit(nopython=True)
def mag(phi):
    #magentization depending on phi
    return np.tanh(beta*h + phi)


@jit(nopython=True)
def energy(phi, N, J):
    #energy depending on phi
    return 1./2./beta/N - phi**2/2./beta**2/J - h*np.tanh(beta*h+phi)

@jit(nopython=True)
def hcm(J, N, Ncfg, Ntherm, Nmd):
    #HCM algorithm: apply first Ntherm times to thermalize system, then Ncfg times to sample configuration
    phi = np.zeros(Ncfg+Ntherm)
    accept = np.zeros(Ncfg+Ntherm)
    #first sample random phi0?
    phi[0] = np.random.normal(0,1)

    for n in range(1, Ncfg+Ntherm):
        p0 = np.random.normal(0,1)
        par = leapfrog(H, p0, phi[n-1], Nmd, N, J)
        if(1 <= np.exp(H(p0, phi[n-1]) - H(*par))): #accept
            phi[n] = par[1]
            accept[n] = 1
        else: phi[n] = phi[n-1]     #reject

    #now take configuration for phi after thermalization
    return phi[Ntherm:], accept[Ntherm:]


def simulate(J, N, Ncfg, Ntherm, Nmd):

    phi_conf, accept = hcm(J, N, Ncfg, Ntherm, Nmd)
    #and calculate m and eps

    m = mag(phi_conf).mean()
    delm = bootstrap_error(mag(phi_conf), 100)

    eps = energy(phi_conf, N, J).mean()
    deleps = bootstrap_error(energy(phi_conf, N, J), 100)

    #acceptance rate:
    acc = accept.mean()

    #exact results:
    def f(J, x):
        return np.exp(0.5*beta*J/N*x**2 + beta*h*x)

    Narray = np.arange(0,N+1,1)
    binom = scipy.special.comb(N, Narray)

    Z = np.sum(binom*f(J, N-2*Narray))
    eps_exact = -1./N/Z*np.sum(binom*(0.5*beta*J/N*(N-2*Narray)**2 + beta*h*(N-2*Narray))*f(J, N-2*Narray)) #minus?
    m_exact = 1./N/Z*np.sum(binom*(N-2*Narray)*f(J, N-2*Narray))

    #missing: acc rate is very low?

    return N, J, m, delm, eps, deleps, acc, m_exact,  eps_exact


Ncfg = 500
Ntherm = 500
Nmd = 1000


N_array = np.arange(5, 21, 1)
Jarray = np.linspace(0.2, 2, 100)

for N in N_array:
    for J in Jarray:
        print("N = {0:d} from 20, J = {1:5.2f} from .2 to 2".format(N,J), end = "\r")
        # write output of the simulate function to a text file
        f = open("results.txt", "a")
        f.write("{0:d}\t{1:e}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\t{6:.4f}\t{7:e}\t{8:e}\n".format(*simulate(J, N, Ncfg, Ntherm, Nmd)))
        f.close()
