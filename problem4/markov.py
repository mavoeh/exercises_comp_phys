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

def generate_markov_chain(Ncfg, Ntherm, Nmd, pars, phi0=0.0):
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
    
# store the markov chain for a specific Nmd in a class
# to make it easier to keep track of the binned data
class mchain:
    """
    markov chain of langth N for long range ising model using
    parameters pars = (N, beta, J, h) and Ntherm thermalization steps
    
    
    the markov chain for the magnetization is stored in self.m
    its autocorrelation in self.autocorr_m
    
    
    the function block(blocksizes) calculates the blocked magnetization
    chains for a list of blocksizes. The blocked lists are stored in
    a dictionary. The individual blocked chains can be accessed by:
    
    self.blocked[b]             blocked chain
    self.blocked_autocorr[b]    autocorrelation of blocked chain
    
    where b is the block size.
    
    
    the bootstrap error estimate of the (un)blocked chains can
    be calculated using the bootstrap method
    """
    def __init__(self, N, Ntherm, Nmd, pars):
        self.N = N
        # generate markov chain and store acceptance, and chain as numpy array
        self.acceptance, self.phi = generate_markov_chain(N, Ntherm, Nmd, pars)
        # calculate magnetization
        self.m = np.tanh(beta*h+self.phi)
        self.autocorr_m = autocorr(self.m)
        print("Generated Markov chain of length %d for "%N, end = '')
        print("N = {0:d}, beta = {1:.1f}, J = {2:.1f} and h = {3:.1f} using Nmd = {4:d}.".format(*pars,Nmd))
        print("mean value of <m>: {0:.3f}\nstandard deviation of <m>: {1:.3f}\n".format(self.m.mean(),self.m.std()))
    
    def block(self, blocksizes):
        self.blocked = {} # create dictionary to store blocked arrays
        self.blocked_autocorr = {} # and their autocorrelations
        for bs in blocksizes:
            blocked = np.zeros(self.N//bs)
            slices = np.arange(0, N, bs, dtype=np.int) # equally spaced array by blocksize
            for index in range(bs):
                blocked += self.m[slices+index] # iteratively calculate sum of blocks
            blocked /= bs # divide by blocksize
            self.blocked[bs] = blocked # add array to dictionary
            self.blocked_autocorr[bs] = autocorr(blocked)
    
    def bootstrap(self, b, nBS):
        """
        performs a bootstrap error estimation, using nBS
        bootstrap samples, of the markov chain with
        blocksize b. If b = 1 use the non-blocked chain.
        
        return the error as float
        """
        if b == 1:
            array = self.m
        else:
            array = self.blocked[b]
        return bootstrap_error(array, nBS)
        


# now that the necessary functions are defined we can start generating and plotting the data

# N = 5 spins, beta = 1, J = 0.1, h = 0.5
n = 5
beta = 1
J = 0.1
h = 0.5
pars = (n, beta, J, h)
Nmds = [4,100]

N = 12800 #length of markov chain
Ntherm = 10000 # more than enough thermalization steps

# create chain classes for the number of mol.dyn. steps in Nmds and store in list
chains = []
for Nmd in Nmds:
    chains.append(mchain(N, Ntherm, Nmd, pars))

# monte carlo time
t = np.linspace(1, N+1, N)



# plot first couple 100 values of m
fig_mhistory = plt.figure()
ax = plt.gca()

tmax = 500 #number of values plotted
colors = ["blue", "red"] # colors used for plotting
for i, Nmd in enumerate(Nmds):
    m = chains[i].m
    ax.plot(t[:tmax], m[:tmax],
            label = ("$N_\mathrm{md} = %d$"%Nmd),
            linestyle = "none",
            marker = "o",
            markersize = 4,
            color = colors[i],
            alpha = 0.5)

ax.set_xlabel("MC time $t$")
ax.set_ylabel("MC history of $\{m\}$")
ax.legend(loc=0)
ax.grid(True)
fig_mhistory.tight_layout()

fig_mhistory.savefig("m_history.pdf")




# now plot autocorrelation functions for the different Nmds
fig_autocorr = plt.figure()
ax = plt.gca()

tmax = 100
for i, Nmd in enumerate(Nmds):
    gamma = chains[i].autocorr_m # store autocorrelation as gamma
    ax.plot(t[:tmax]-1, gamma[:tmax],
            label = ("$N_\mathrm{md} = %d$"%Nmd),
            linestyle = "none",
            marker = "o",
            color = colors[i],
            markersize = 5)

ax.set_xlabel(r"Time $\tau$")
ax.set_ylabel(r"Normalized autocorrelation $C(\tau)$")
ax.legend(loc=0)
ax.grid(True)
fig_autocorr.tight_layout()

fig_autocorr.savefig("m_autocorr.pdf")


# from now on work with the Nmd = 100 data
chain = chains[1]

# calculate blocked arrays
bs = [2,4,8,16,32,64]
chain.block(bs)


# plot autocorrelations of blocked lists
tmax = 25
fig_blocked_autocorrs = plt.figure()
ax = plt.gca()
for i, b in enumerate(bs):
    ax.plot(t[:tmax]-1, chain.blocked_autocorr[b][:tmax],
            label = ("b = %d"%b),
            linestyle = ":",
            linewidth = 1,
            marker = "o",
            markersize = 3)
    
ax.set_xlabel(r"Time $\tau$")
ax.set_ylabel(r"Normalized autocorrelation $C(\tau)$")
ax.legend(loc=0)
ax.grid(True)
fig_blocked_autocorrs.tight_layout()

fig_blocked_autocorrs.savefig("blocked_autocorrs.pdf")


# plot naive standard error
fig_naive_error = plt.figure()
ax = plt.gca()
sd = np.zeros(len(bs)+1)
sd[0] = chain.m.std() / np.sqrt(chain.N)
for i, b in enumerate(bs):
    sd[i+1] = chain.blocked[b].std() / np.sqrt(chain.N/b)
ax.plot([1]+bs, sd, marker = "o",
        linestyle = "none",
        markersize = 3)

ax.set_xlabel("Blocksize $b$")
ax.set_ylabel(r"Naive standard error $\sigma / \sqrt{N/b}$")
ax.grid(True)
fig_naive_error.tight_layout()

fig_naive_error.savefig("naive_standard_error.pdf")


# plot nBS dependence of bootstrap error for all blocksizes
fig_nbs_dependence = plt.figure()
ax = plt.gca()
nBS = np.arange(10, 301, 1)
for i, b in enumerate([1]+bs):
    error = np.zeros(len(nBS))
    for j, n in enumerate(nBS):
        error[j] = chain.bootstrap(b, n)
    ax.plot(nBS, error, label = ("b = %d"%b))

ax.set_xlabel(r"Number of bootstrap samples $N_\mathrm{bs}$")
ax.set_ylabel(r"Bootstrap error estimate $\delta m$")
ax.legend(loc=0)
ax.grid(True)
fig_nbs_dependence.tight_layout()

fig_nbs_dependence.savefig("nbs_dependence.pdf")
