import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit


beta = 1    #beta = 1/kB/T
h = 0.5     #coupling to extenal magnet field

@jit(nopython=True)
def H(p, phi):
    return p**2/2 + phi**2/(2*beta*J/N) - N*np.log(2*np.cosh(beta*h+phi))

@jit(nopython=True)
def leapfrog(p, phi, Nmd, N, J):
    eps = 1/Nmd
    phi += eps/2 * p
    for n in range(Nmd-1):
        p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
        phi += eps*p
    p -= eps*(phi*N/(beta*J) - N*np.tanh(beta*h+phi))
    phi += eps/2*p
    return p, phi


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
    phi[0] = 0

    for n in range(1, Ncfg+Ntherm+1):
        p0 = np.random.normal(0,1)
        par = leapfrog(p0, phi[n-1], Nmd, N, J)
        if(np.random.uniform(0,1) <= np.exp(H(p0, phi[n-1]) - H(*par))): #accept
            #print(np.exp(H(p0, phi[n-1]) - H(*par)))
            phi[n] = par[1]
            accept[n] = 1
        else: 
            phi[n] = phi[n-1]     #reject
            #print(np.exp(H(p0, phi[n-1]) - H(*par)))

    #now take configuration for phi after thermalization
    return phi[Ntherm:], accept[Ntherm:]


def simulate(J, N, Ncfg, Ntherm, Nmd):

    phi_conf, accept = hcm(J, N, Ncfg, Ntherm, Nmd)
    #and calculate m and eps
    #print(phi_conf)

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
Ntherm = 1000
Nmd = 2000

N = 5
steps = 20
J_array = np.linspace(0.2,2,steps)

eps = np.zeros(steps)
eps_exact = eps.copy()
deleps = eps.copy()
m = eps.copy()
m_exact = eps.copy()
delm = eps.copy()
acc = eps.copy()
for i in range(steps):
    J = J_array[i]
    _, _, m[i], delm[i], eps[i], deleps[i], acc[i], m_exact[i], eps_exact[i] = simulate(J, N, Ncfg, Ntherm, Nmd)

plt.figure()
plt.errorbar(J_array, m, delm, linestyle="None")
plt.plot(J_array, m_exact)
plt.scatter(J_array, acc)
plt.show()
    

"""
N_array = np.arange(5, 21, 1)
Jarray = np.linspace(0.2, 2, 100)

for N in N_array:
    for J in Jarray:
        print("N = {0:d} from 20, J = {1:5.2f} from .2 to 2".format(N,J), end = "\r")
        # write output of the simulate function to a text file
        f = open("results.txt", "a")
        f.write("{0:d}\t{1:e}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\t{6:.4f}\t{7:e}\t{8:e}\n".format(*simulate(J, N, Ncfg, Ntherm, Nmd)))
        f.close()
"""
