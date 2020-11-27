import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from numba import jit
from leapfrog import H, leapfrog

@jit(nopython=True)
def metropolis_hastings_step(H, phi0, Nmd, pars):
    # perform the hmc metropolis hastings step
    p0 = np.random.normal(0,1)
    p, phi = leapfrog(p0, phi0, Nmd, pars)
    if np.random.uniform(0,1) <= np.exp(H(p0,phi0,pars)-H(p,phi,pars)):
        # if accepted, return tuple with boolean value true if accepted
        return True, phi # and new phi
    else:
        return False, phi0 # if rejected return False and old phi

def markov_chain(Ncfg, Ntherm, Nmd, pars, phi0=0.0):
    """
    generate markov chain of length Ncfg using the metropolis hastings
    accept reject for the long range ising model
    
    start at phi0 doing Ntherm thermalization accept/reject
    steps without storing anything
    
    use parameters pars = (N, beta, J, h)
    
    returns: 
    (acceptance rate, markov chain)
    tuple containing the acceptance rate and the markov chain
    for the values of phi
    """
    
    # assign parameters
    N, beta, J, h = pars
    
    # thermalize
    for i in range(Ntherm):
        _, phi0 = metropolis_hastings_step(H, phi0, Nmd, pars)
    
    # initialize array for phi values
    phi_array = np.zeros(Ncfg)
    counter = 0 # counter for acceptance rate
    for i in range(Ncfg):
        accepted, phi_array[i] = metropolis_hastings_step(H, phi0, Nmd, pars)
        phi0 = phi_array[i]
        if accepted:
            counter += 1
    
    counter /= Ncfg # normalize counter
    
    return counter, phi_array # return acceptance rate and markov_chain
    
# N = 5 spins, beta = 1, J = 1, h = 0.5
n = 5
beta = 1
J = 1
h = 0.5
pars = (n, beta, J, h)

N = 12800 #length of markov chain
Ntherm = 10000 # more than enough thermalization steps

# create markov chains
acceptance_nmd4, phi_nmd4 = markov_chain(N, Ntherm, 4, pars)
acceptance_nmd100, phi_nmd100 = markov_chain(N, Ntherm, 100, pars)

# calculate magnetizationspytho
m_nmd4 = np.tanh(beta*h+phi_nmd4)
m_nmd100 = np.tanh(beta*h+phi_nmd100)

# monte carlo time
t = np.linspace(1, N+1, N)

# plot first couple 100 values of m
fig_mhistory = plt.figure()
ax = plt.gca()

tmax = 200 #number of values plotted
ax.plot(t[:tmax], m_nmd4[:tmax],
        label = "$N_\mathrm{md} = 4$",
        linestyle = "none",
        marker = "o",
        markersize = 4,
        color = "blue",
        alpha = 0.5)

ax.plot(t[:tmax], m_nmd100[:tmax],
        label = "$N_\mathrm{md} = 100$",
        linestyle = "none",
        marker = "o",
        markersize = 4,
        color = "red",
        alpha = 0.5)

ax.set_xlabel("MC time $t$")
ax.set_ylabel("MC history of $\{m\}$")
ax.legend(loc=0)
ax.grid(True)
fig_mhistory.tight_layout()

fig_mhistory.savefig("m_history.pdf")

# estimate for m
avg_m_nmd4 = m_nmd4.mean()
avg_m_nmd100 = m_nmd100.mean()

print(avg_m_nmd4, avg_m_nmd100)
