#sys.settrace
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from numba import jit



@jit(nopython=True)
# for finest grid: phi = 0
def H(u, a, phi):
    Ham = 1./a*np.sum((u[1:] - u[:-1])**2) + a*np.sum(phi*u)
    return Ham

#analytic solution for expectation value of m^2
@jit(nopython=True)
def m_squared(pars):
    N, beta, a, delta = pars
    k = np.arange(1, N, 1)
    return 1./2.*(a/4./N/beta)*np.sum(1./np.sin(k*np.pi/2./N)**2)

#analytic solution for expectation value of energy
@jit(nopython=True)
def energy(pars):
    N, beta, a, delta = pars
    return (N-1)/2./beta



@jit(nopython=True)
def metropolis_hastings_step(u, delta, a, phi):
    # perform the hmc metropolis hastings step
    r = np.random.uniform(-1,1)
    x = np.random.choice(np.arange(1, N-1, 1)) #boundary conditions u(0) = u(N)= 0
    change = delta*r 
    DeltaH = 2./a*(u[x]**2 - (u[x]+change)**2 + change*u[x-1] + change*u[x+1]) - a*phi[x]*change
    if np.random.uniform(0,1) <= np.exp(DeltaH):
        # if accepted, return tuple with boolean value true if accepted
        u[x] += change
        return True, u # and new u
    else:
        return False, u # if rejected return False and old u

@jit(nopython=True)
def sweep(u, pars, phi):
    N, beta, a, delta = pars
    counter = 0
    for i in range(N-1):
        c, u = metropolis_hastings_step(u, delta, a, phi)
        if c:
            counter += 1.
    return counter, u


def generate_markov_chain(Ncfg, Ntherm, pars, u0, phi):
    """
    generate markov chain of length Ncfg using the metropolis hastings
    accept reject N-1 times (sweep) for the gaussian model
    
    start at config u0 doing Ntherm sweeps without storing anything
    
    use parameters pars = (N, beta, delta, a) and phi for external field
    
    returns: 
    (acceptance rate, markov chain)
    tuple containing the acceptance rate and the markov chain
    for the arrays u
    """
    
    # assign parameters
    N, beta, a, delta = pars
    
    # thermalize
    for i in range(Ntherm):
        _, u0 = sweep(u0, pars, phi)

    # initialize list for u arrays and arrays to store measurement for m and energy
    u_list = []
    counter = 0 # counter for acceptance rate
    for i in range(Ncfg):
        accepted, u = sweep(u0, pars, phi)
        u_list.append(list(u))
        u0 = np.array(u_list[i])
        counter += accepted
    
    counter /= Ncfg*N # normalize counter
    
    return counter, u_list # return acceptance rate and markov_chain


# only works if N is even -> then we have N+1 components in u
def fine_to_coarse(u):
    u = u[::2]
    u[0] = 0
    u[-1] = 0
    return u

def coarse_to_fine(u):
    u_fine = np.zeros(len(u)*2-1)
    for i in range(len(u_fine)):
        if i%2==0:
            u_fine[i] = u[i//2]
        else:
            u_fine[i] = (u[(i-1)//2] + u[(i+1)//2])/2.

    u_fine[0] = 0
    u_fine[-1] = 0

    return u_fine


def multigrid(pre, post, n, gamma, pars, u0, phi):
    '''
    Multigrid simulation algorithm in order to update u properly and then make a measurement for an observale
    Input Paramters:
        pre     number of sweeps pre-coarsening, array with number for each level 
        post    number of sweeps post-coarsening, array with number for each level 
        n       number of levels 
        gamma   number of multigrid cycles in step 
        u       initial array on fine grid

    Output: array u for measurment 
    '''

    N, beta, a, delta = pars

    #step1: pre pre-coarsening sweeps (i n==1, then coarsest level is reached -> skip to step 5)
    if n != 1:
        print("pre coarse sweep", n)
        for j in range(pre[n-1]):
            accepted, u = sweep(u0, pars, phi)
            u0 = u

    #step 2: coarsening
        print("coarsening to", n-1)
        u_coarse = fine_to_coarse(u)
        N = N//2
        a *= 2
        pars = (N, beta, a, delta)
        phi_coarse = np.zeros(len(u_coarse))

        for i in range(1, len(u_coarse)-1):
            phi_coarse[i] = 1./4.*(phi[2*i+1]+2*phi[i]+phi[2*i-1]) + 1./2./a**2*(2*u[i] - u[2*i+2] - u[2*i-2])

        n -= 1

    #step 3: recursion gamma times
        print("recursion", n)
        for g in range(gamma):
            uc = multigrid(pre, post, n, gamma, pars, u_coarse, phi_coarse)

        n += 1
    #step 4: updating current u(a)
        print("prolongation to", n)
        u += coarse_to_fine(uc)

    #step 5: post post-prolongation sweeps at current level
    print("post sweep", n)
    for k in range(post[n-1]):
        accepted, u = sweep(u0, pars, phi)
        u0 = u

    return u

'''
#test markokv chain algorithm

N = 64
beta = 1.
a = 1.
delta = 2.
pars = (N, beta, a, delta)
phi = np.zeros(N+1)

Ncfg = 10000
Ntherm = 100000

u0 = np.zeros(N+1)

counter, u_list  = generate_markov_chain(Ncfg, Ntherm, pars, u0, phi)

#calculate mean value for m and energy
m_array = np.zeros(Ncfg)
eps_array = np.zeros(Ncfg)
m2_array = np.zeros(Ncfg)

#plot history of m and eps
for i in range(Ncfg):
    m_array[i] = 1./N*np.array(u_list[i]).mean()
    eps_array[i] = H(np.array(u_list[i]), a, phi)
    m2_array[i] = 1./N*(np.array(u_list[i])**2).mean()

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
delm = m_array.std()
m2 = m2_array.mean()
delm2 = m2_array.std()
eps = eps_array.mean()
deleps = eps_array.std()


m2exact = m_squared(pars)
epsexact = energy(pars) 

print(m, delm, "exact:", 0)
print(m2, delm2, "exact:", m2exact)
print(eps, deleps, "exact:", epsexact)
'''

#test multigrid
N = 64
beta = 1.
a = 1
delta = 2.
phi = np.zeros(N+1)

pars = (N, beta, a, delta)

n = 3
pre = [4, 2, 1]
post = [4, 2, 1]

#first for gamma = 1
gamma = 1

u = np.zeros(N+1)

unew = multigrid(pre, post, n, gamma, pars, u, phi)

#print(unew)
