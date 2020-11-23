import numpy as np
import matplotlib.pyplot as plt


beta = 1    #beta = 1/kB/T
h = 0.5     #coupling to extenal magnet field

def H(p, phi):
    return p**2/2 + phi**2/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))

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

def mag(phi):
    #magentization depending on phi
    return np.tanh(beta*h + phi)

def energy(phi, N, J):
    #energy depending on phi
    return 1./2./beta/N + phi**2/2./beta**2/J - h*np.tanh(beta*h+phi)


def simulate(J, N, Ncfg, Ntherm, Nmd):

    #HCM algorithm: apply first Ntherm times to thermalize system, then Ncfg times to sample configuration
    phi = np.zeros(Ncfg+Ntherm)
    accept = np.zeros(Ncfg+Ntherm)
    #first sample random phi0
    phi[0] = 1
    accept[0] = 0

    for n in range(1, Ncfg+Ntherm):
        p0 = np.random.sample()

        par = leapfrog(H, p0, phi[n-1], Nmd, N, J)
        if np.exp(H(p0, phi[n-1]) - H(*par)) < 1:
            phi[n] = par[1]
            accept[n] += 1
        else: phi[n] = phi[n-1]


    #now take configuration for phi after thermalization

    phi_conf = phi[Ntherm:]

    #and calculate m and eps

    m = np.sum(mag(phi_conf))/len(phi_conf)  
    eps = np.sum(energy(phi_conf, N, J))/len(phi_conf)

    #acceptance rate:
    acc = np.sum(accept)/len(accept)

    #missing: error calc and exact result, acc rate is very low?

    return N, J, m, eps, acc


J = 1
N = 10
Ncfg = 1000
Ntherm = 1000
Nmd = 5000

results = simulate(J, N, Ncfg, Ntherm, Nmd)

print(results)