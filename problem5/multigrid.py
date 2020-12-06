import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from numba import jit



@jit(nopython=True)
def H(u, a):
    Ham = np.sum((u[1:] - u[:-1])**2)
    return 1./a*Ham


@jit(nopython=True)
def metropolis_hastings_step(u, delta, a):
    # perform the hmc metropolis hastings step
    r = np.random.uniform(-1,1)
    x = np.random.choice(np.arange(1, N-1, 1)) #boundary conditions u(0) = u(N)= 0
    change = delta*r 
    DeltaH = 2./a*(u[x]**2 - (u[x]+change)**2 + change*u[x-1] + change*u[x+1])
    if np.random.uniform(0,1) <= np.exp(DeltaH):
        # if accepted, return tuple with boolean value true if accepted
        u[x] += change
        return True, u # and new u
    else:
        return False, u # if rejected return False and old u

@jit(nopython=True)
def sweep(u, pars):
    N, beta, a, delta = pars
    counter = 0
    for i in range(N-1):
        c, u = metropolis_hastings_step(u, delta, a)
        if c:
            counter += 1.
    return counter, u


def generate_markov_chain(Ncfg, Ntherm, pars, u0):
    """
    generate markov chain of length Ncfg using the metropolis hastings
    accept reject N-1 times (sweep) for the gaussian model
    
    start at config u0 doing Ntherm sweeps without storing anything
    
    use parameters pars = (N, beta, delta, a)
    
    returns: 
    (acceptance rate, markov chain)
    tuple containing the acceptance rate and the markov chain
    for the arrays u
    """
    
    # assign parameters
    N, beta, a, delta = pars
    
    # thermalize
    for i in range(Ntherm):
        _, u0 = sweep(u0, pars)

    # initialize list for u arrays and arrays to store measurement for m and energy
    u_list = []
    #m_array = np.zeros(Ncfg)
    #eps_array = np.zeros(Ncfg)
    counter = 0 # counter for acceptance rate
    for i in range(Ncfg):
        accepted, u = sweep(u0, pars)
        u_list.append(list(u))
        u0 = np.array(u_list[i])
        #m_array[i] = np.array(u_list[i]).mean()
        #eps_array[i] = H(np.array(u_list[i]), a)

        counter += accepted
    
    counter /= Ncfg*(N-1) # normalize counter
    
    return counter, u_list # return acceptance rate and markov_chain



#test markokv chain algorithm

N = 64
beta = 1.
a = 1.
delta = 2.
pars = (N, beta, a, delta)

Ncfg = 10000
Ntherm = 1000000

u0 = np.zeros(N)

counter, u_list  = generate_markov_chain(Ncfg, Ntherm, pars, u0)


#calculate mean value for m and energy
m_array = np.zeros(Ncfg)
eps_array = np.zeros(Ncfg)

#plot history of m and eps
for i in range(Ncfg):
    m_array[i] = np.array(u_list[i]).mean()
    eps_array[i] = H(np.array(u_list[i]), a)


fig_mhistory = plt.figure()
ax = plt.gca()

ax.plot(np.arange(0, Ncfg, 1), m_array,
            linestyle = "none",
            marker = "o",
            markersize = 4,
            alpha = 0.5)

ax.set_xlabel("MC time $t$")
ax.set_ylabel("MC history of $\{m\}$")
ax.grid(True)
fig_mhistory.tight_layout()

fig_mhistory.savefig("m_history.pdf")

m = m_array.mean()
eps = eps_array.mean()

print(m, eps)