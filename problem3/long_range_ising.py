import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit
from leapfrog import H, leapfrog

@jit(nopython=True)
def metropolis_hastings_step(H, phi0, pars):
    # perform the hmc metropolis hastings step
    p0 = np.random.normal(0,1)
    p, phi = leapfrog(p0, phi0, Nmd, pars)
    if np.random.uniform(0,1) <= np.exp(H(p0,phi0,pars)-H(p,phi,pars)):
        # if accepted, return tuple with boolean value true if accepted
        return True, phi # and new phi
    else:
        return False, phi0 # if rejected return False and old phi

def bootstrap_error(array, nBS):
    # performs a bootstrap error estimation for an array, generating nBS bootstrap samples  
    n = len(array)
    bsmean = np.zeros(nBS)
    for i in range(nBS):
        indices = np.random.randint(n,size=n) # random bootstrap indices
        bsmean[i] = array[indices].mean()
    return bsmean.std()

def simulate(Ncfg, Ntherm, Nmd, nBS, pars, phi0=0.0):
    
    # assign parameters
    N, beta, J, h = pars
    
    # thermalize
    for i in range(Ntherm):
        _, phi0 = metropolis_hastings_step(H, phi0, pars)
    
    # initialize array for phi values
    phi_array = np.zeros(Ncfg)
    counter = 0 # counter for acceptance rate
    for i in range(Ncfg):
        accepted, phi_array[i] = metropolis_hastings_step(H, phi0, pars)
        phi0 = phi_array[i]
        if accepted:
            counter += 1
    
    counter /= Ncfg # normalize counter
    
    # assign value and error of <m>
    m = np.tanh(beta*h+phi_array)
    dm = bootstrap_error(m, nBS)
    m = np.mean(m)
    
    # assign value and error of epsilon
    eps = 1/(2*beta*N) - phi_array**2/(2*J*beta**2) - h*np.tanh(beta*h+phi_array)
    deps = bootstrap_error(eps, nBS)
    eps = np.mean(eps)
    
    return (m, dm, eps, deps, counter) , phi0 # return results and last phi (to be used as start in next run)

#exact results:
@jit(nopython=True)
def f(pars, x):
    N, beta, J, h = pars
    return np.exp(0.5*beta*J/N*x**2 + beta*h*x)

def Z(pars):
    N, beta, J, h = pars
    Narray = np.arange(0, N+1, 1)
    binom = scipy.special.comb(N, Narray)
    return np.sum(binom*f(pars, N-2*Narray))

def eps_exact(pars):
    N, beta, J, h = pars
    Narray = np.arange(0, N+1, 1)
    binom = scipy.special.comb(N, Narray)
    return -1./N/Z(pars)*np.sum(binom*(0.5*beta*J/N*(N-2*Narray)**2 + beta*h*(N-2*Narray))*f(pars,N-2*Narray))

def m_exact(pars):
    N, beta, J, h = pars
    Narray = np.arange(0, N+1, 1)
    binom = scipy.special.comb(N, Narray)
    return 1./N/Z(pars)*np.sum(binom*(N-2*Narray)*f(pars,N-2*Narray))
    

def Jplot(Ncfg, Ntherm, Nmd, nBS, N_values, J_array):
    """
    plots <m> for 4 values of N contained in the list N_values
    vs the values of J cointained in J_array
    uses external coupling h = 0.5 and beta = 1
    
    returns:
    singlefig          pyplot figure in which all plots are overlaid
    multifig_m         pyplot figure with 4 subplots
    """
    font = {"fontname":"Times New Roman", "fontsize":18}
    N = None
    beta = 1.
    J = None
    h = 0.5
    pars = [N, beta, J, h]
    steps = len(J_array)
    
    # array to which the data (value and error for m and eps) is saved
    data = np.zeros((steps, 5))
    
    # array for J values with lower stepsize for analytic solution
    J_fine = np.linspace(J_array.min(),J_array.max(),200)
    
    # create figs/axes for m absolute plot
    multifig_m, axs_m = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,6))
    multifig_m.text(0.535, 0.02, "$J$", ha='center', **font)
    multifig_m.text(0.02, 0.525, r'$\langle m \rangle$', va='center', rotation='vertical', **font)
    singlefig_m, singleax_m = plt.subplots(1,1,figsize=(8,6))
    
    #create figs/axes for epsilon plot
    multifig_eps, axs_eps = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,6))
    multifig_eps.text(0.515, 0.02, "$J$", ha='center', **font)
    multifig_eps.text(0.02, 0.505, '$\epsilon$', va='center', rotation='vertical', **font)
    singlefig_eps, singleax_eps = plt.subplots(1,1,figsize=(8,6))
    
    colors = ['magenta','lime','red','blue']
    for i, N in enumerate(N_values):
        pars[0] = N
        
        # first do some initial thermalization to determine a good starting value for phi
        pars[2] = J_array[0]
        phi0 = 0
        for _ in range(50*Ntherm): #50*Ntherm thermalization steps
            _, phi0 = metropolis_hastings_step(H, phi0, pars)
        # numeric solutions
        for j, J in enumerate(J_array):
            pars[2] = J
            print("N = {0:.0f}, J = {1:.2f}, phi0 = {2:.2f}".format(pars[0],pars[2],phi0))
            data[j,:], phi0 = simulate (Ncfg, Ntherm, Nmd, nBS, pars, phi0)
            phi0 += np.random.normal(-0.25,0.25) # use last value of phi0 as new starting value
            # but jiggle it a little bit
            
        # analytic solutions
        m_ex = np.zeros(200)
        eps_ex = m_ex.copy()
        for j, J in enumerate(J_fine):
            pars[2] = J
            m_ex[j] = m_exact(pars)
            eps_ex[j] = eps_exact(pars)
        # plot m
        ax_m = axs_m[i//2,i%2]
        ax_m.errorbar(J_array, data[:,0], data[:,1],
                      label = ("N = %d"%N),
                      linestyle = "none",
                      capsize = 2,
                      capthick = 2)
        ax_m.plot(J_fine, m_ex, c="black", label="analytic")
        ax_m.legend(loc=4)
        ax_m.grid(True)
        ax_m.set_xlim(min(J_array)-0.02,max(J_array)+0.02)
        ax_m.set_ylim(min(data[:,0])-0.02,max(data[:,0])+0.02)
        singleax_m.scatter(J_array, data[:,0],
                           s = 30,
                           edgecolor="none",
                           c = colors[i],
                           label = ("N = %d"%N))
        singleax_m.set_xlim(min(J_array)-0.02,max(J_array)+0.02)
        singleax_m.set_ylim(min(data[:,0])-0.02,max(data[:,0])+0.02)
        singleax_m.set_xlabel("$J$", **font)
        singleax_m.set_ylabel(r'$\langle m \rangle$', **font)
        singleax_m.grid(True)
        singleax_m.legend(loc=4)
        # plot epsilon
        ax_eps = axs_eps[i//2,i%2]
        ax_eps.errorbar(J_array, data[:,2], data[:,3],
                        label = ("N = %d"%N),
                        linestyle = "none",
                        capsize = 2,
                        capthick = 2)
        ax_eps.plot(J_fine, eps_ex, c="black", label="analytic")
        ax_eps.legend(loc=3)
        ax_eps.grid(True)
        ax_eps.set_xlim(min(J_array)-0.02,max(J_array)+0.02)
        ax_eps.set_ylim(min(data[:,2])-0.02,max(data[:,2])+0.02)
        singleax_eps.scatter(J_array, data[:,2],
                             s = 30,
                             edgecolor="none",
                             c = colors[i],
                             label = ("N = %d"%N))
        singleax_eps.set_xlim(min(J_array)-0.02,max(J_array)+0.02)
        singleax_eps.set_ylim(min(data[:,2])-0.02,max(data[:,2])+0.02)
        singleax_eps.set_xlabel("$J$", **font)
        singleax_eps.set_ylabel(r'$\epsilon$', **font)
        singleax_eps.grid(True)
        singleax_eps.legend(loc=3)
        #analytic solutions
        singleax_m.plot(J_fine, m_ex, c=colors[i])
        singleax_eps.plot(J_fine, eps_ex, c=colors[i])
        multifig_m.tight_layout(rect=[0.04, 0.04, 1, 1])
    return singlefig_m, multifig_m, singlefig_eps, multifig_eps



Ncfg = 500
Ntherm = 1000
Nmd = 100
nBS = 200
J_array = np.linspace(0.2,2,19)
N_list = [5.,10.,15.,20.] # hast to be floats (otherwise numba raises an error)
singlefig_m, multifig_m, singlefig_eps, multifig_eps = Jplot(Ncfg, Ntherm, Nmd, nBS, N_list, J_array)
singlefig_m.savefig("singlefig_m.pdf")
multifig_m.savefig("multifig_m.pdf")
singlefig_eps.savefig("singlefig_eps.pdf")
multifig_eps.savefig("multifig_eps.pdf")
