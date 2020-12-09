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
    return a/8./N/beta*np.sum(1./np.sin(k*np.pi/2./N)**2)

#analytic solution for expectation value of energy
@jit(nopython=True)
def energy(pars):
    N, beta, a, delta = pars
    return (N-1)/2./beta


@jit(nopython=True)
def metropolis_hastings_step(u, pars, phi):
    N, beta, a, delta = pars
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
    #do metropolis hastings step N-1 times
    for i in range(N-1):
        c, u = metropolis_hastings_step(u, pars, phi)
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
@jit(nopython=True)
def fine_to_coarse(u):
    return u[::2]

@jit(nopython=True)
def coarse_to_fine(u):
    u_fine = np.zeros(len(u)*2-1)
    for i in range(len(u_fine)):
        if i%2==0:
            u_fine[i] = u[i//2]
        else:
            u_fine[i] = (u[(i-1)//2] + u[(i+1)//2])/2.

    return u_fine

@jit(nopython=True)
def autocorr(x):
    """
    calculates the autocorrelation function of a 1d-numpy array x
    """
    N = len(x)
    mean = x.mean()
    gamma = np.zeros(N) # array in which the autocorrelation is stored
    gamma[0] = 1/N * np.sum( (x-mean)*(x-mean) )
    for tau in range(1,N):
        gamma[tau] = 1 / (N-tau) * np.sum( (x[:-tau]-mean) * (x[tau:]-mean) )
    return gamma/gamma[0]

def bootstrap_error(array, nBS):
    # performs a bootstrap error estimation for an array, generating nBS bootstrap samples  
    n = len(array)
    bsmean = np.zeros(nBS)
    for i in range(nBS):
        indices = np.random.randint(n,size=n) # random bootstrap indices
        bsmean[i] = array[indices].mean()
    return bsmean.std()


def multigrid(pre, post, n, gamma, pars, u0, phi):
    '''
    Multigrid simulation algorithm in order to update u properly and then make a measurement for an observale
    Input Paramters:
        pre     number of sweeps pre-coarsening, array with number for each level 
        post    number of sweeps post-coarsening, array with number for each level 
        n       number of levels (initially finest level at n, coarsest at n=1)
        gamma   number of multigrid cycles in step 
        u       initial array on fine grid
        phi     external field = 0 initially

    Output: array u for measurment 
    '''
    N, beta, a, delta = pars

    #step 1: pre coarsening sweeps, if NOT at the coarsest level (n = 1) 
    if n > 1:
        #print("pre sweeps", n)
        for k in range(pre[n-1]):
            _, u0 = sweep(u0, pars, phi) 

    #step 2: coarseing to next coarser level
        #print("coarseing to", n-1)        
        u_coarse = np.zeros(N//2 +1)
        phi_coarse = np.zeros(N//2 +1)
        for i in range(1, len(phi_coarse)-1):
            phi_coarse[i] = 1./4.*(phi[2*i+1] + 2*phi[i] + phi[2*i-1]) + 1./2./a**2*(2*u0[i] - u0[2*i+2]-u0[2*i-2])

        N = N//2
        a = 2*a
        pars = (N, beta, a, delta)
        n -= 1

    #step 3: recursive step, gamma times
        #print("recusive step at", n)
        for g in range(gamma):
            u_coarse = multigrid(pre, post, n, gamma, pars, u_coarse, phi_coarse)

    #step 4: prolongation & correction to current level
        #print("correction at", n+1)
        u0 += coarse_to_fine(u_coarse)

    #step 5: post correction sweeps
    #print("post sweeps", n+1)
    for j in range(post[n-1]):
        _, u0 = sweep(u0, pars, phi)

    return u0    



#test markokv chain algorithm

N = 64
beta = 1.
a = 1.
delta = 2.
pars = (N, beta, a, delta)
phi = np.zeros(N+1)

Ncfg = 1000
Ntherm = 1000

u0 = np.zeros(N+1)

counter, u_list  = generate_markov_chain(Ncfg, Ntherm, pars, u0, phi)

#calculate mean value for m and energy
m_array = np.zeros(Ncfg)
eps_array = np.zeros(Ncfg)
m2_array = np.zeros(Ncfg)


for i in range(Ncfg):
    m_array[i] = 1./N*np.array(u_list[i]).mean()
    eps_array[i] = H(np.array(u_list[i]), a, phi)
    m2_array[i] = 1./N*(np.array(u_list[i])**2).mean()

#plot history of m and eps
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


#test multigrid algorithm


N = 64
beta = 1.
a = 1
delta = 2.
phi = np.zeros(N+1)

pars = (N, beta, a, delta)

n = 3
pre = [4, 2, 1]
post = [4, 2, 1]

m2exact = m_squared(pars)
epsexact = energy(pars) 

Nmeas = 100 #number of measurements

#first for gamma = 1
gamma = 1

#calculate mean value for m and energy
m_array = np.zeros(Nmeas)
eps_array = np.zeros(Nmeas)
m2_array = np.zeros(Nmeas)


u = np.zeros(N+1)
for i in range(Nmeas):
    u = multigrid(pre, post, n, gamma, pars, u, phi)
    m_array[i] = 1./N*u.mean()
    eps_array[i] = H(u, a, phi)
    m2_array[i] = 1./N*(u**2).mean()


m = m_array.mean()
delm = bootstrap_error(m_array,100)
m2 = m2_array.mean()
delm2 = bootstrap_error(m2_array,100)
eps = eps_array.mean()
deleps = bootstrap_error(eps_array,100)


print(m, delm, "exact:", 0)
print(m2, delm2, "exact:", m2exact)
print(eps, deleps, "exact:", epsexact)

#plot autocorr of m^2
fig_m2corr = plt.figure()
ax = plt.gca()

ax.plot(np.arange(0, Nmeas, 1), autocorr(m2_array),
            linestyle = "none",
            marker = "o",
            markersize = 4,
            alpha = 0.5)

ax.set_xlabel("MC time $t$")
ax.set_ylabel("Autocorrelation of $\{m^2\}$")
ax.grid(True)
fig_m2corr.tight_layout()

fig_m2corr.savefig("m2corr_1.pdf")


#now for gamma = 2
gamma = 2

#calculate mean value for m and energy
m_array = np.zeros(Nmeas)
eps_array = np.zeros(Nmeas)
m2_array = np.zeros(Nmeas)


u = np.zeros(N+1)
for i in range(Nmeas):
    u = multigrid(pre, post, n, gamma, pars, u, phi)
    m_array[i] = 1./N*u.mean()
    eps_array[i] = H(u, a, phi)
    m2_array[i] = 1./N*(u**2).mean()


m = m_array.mean()
delm = bootstrap_error(m_array,100)
m2 = m2_array.mean()
delm2 = bootstrap_error(m2_array,100)
eps = eps_array.mean()
deleps = bootstrap_error(eps_array,100)

print(m, delm, "exact:", 0)
print(m2, delm2, "exact:", m2exact)
print(eps, deleps, "exact:", epsexact)


#plot autocorrelation of m^2

fig_m2corr = plt.figure()
ax = plt.gca()

ax.plot(np.arange(0, Nmeas, 1), autocorr(m2_array),
            linestyle = "none",
            marker = "o",
            markersize = 4,
            alpha = 0.5)

ax.set_xlabel("MC time $t$")
ax.set_ylabel("Autocorrelation of $\{m^2\}$")
ax.grid(True)
fig_m2corr.tight_layout()

fig_m2corr.savefig("m2corr_2.pdf")


### There is some mistake in the algorithm: the results are some orders of magnitude wrong