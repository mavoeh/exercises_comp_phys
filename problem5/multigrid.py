#sys.settrace
import numpy as np
import scipy.special
import matplotlib.pyplot as plt



# for finest grid: phi = 0
def H(u, a, phi):
    Ham = 1./a*np.sum((u[1:] - u[:-1])**2) + a*np.sum(phi*u)
    return Ham


def metropolis_hastings_step(u, phi, pars):
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

def sweep(u, pars, phi):
    N, beta, a, delta = pars
    counter = 0
    for i in range(N-1):
        c, u = metropolis_hastings_step(u, phi, pars)
        if c:
            counter += 1.
    return counter, u


# only works if N is even -> then we have N+1 components in u
def fine_to_coarse(u):
    return u[::2]

def coarse_to_fine(u):
    u_fine = np.zeros(len(u)*2-1)
    l = len(u_fine)
    print ( "# [coarse_to_fine] l = ", l )
    return u_fine
#    for i in range(len(u_fine)):
#        if i%2==0:
#            u_fine[i] = u[i//2]
#        else:
#            u_fine[i] = (u[(i-1)//2] + u[(i+1)//2])/2.
#
#    return u_fine



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
    print("pre sweeps at ", n, ", nu = ", pre[n-1])
    for k in range(pre[n-1]):
        _, u0 = sweep(u0, pars, phi) 

    if n > 1:
    #step 2: coarseing to next coarser level
        print("coarseing ", n, " to ", n-1)        
        u_coarse = np.zeros(N//2 +1)
        phi_coarse = np.zeros(N//2 +1)
        for i in range(1, len(phi_coarse)-1):
            phi_coarse[i] = 1./4.*(phi[2*i+1] + 2*phi[i] + phi[2*i-1]) + 1./2./a**2*(2*u0[i] - u0[2*i+2]-u0[2*i-2])

        # coarse level parameters
        pars_coarse = (N//2, beta, 2*a, delta)

    #step 3: recursive step, gamma times
        print("recusive step at ", n, " to ", n-1)
        for g in range(gamma):
            u_coarse = multigrid(pre, post, n-1, gamma, pars_coarse, u_coarse, phi_coarse)


    #step 4: prolongation & correction to current level
        print("coarse field correction at ", n )
        u0 += coarse_to_fine(u_coarse)

    #step 5: post correction sweeps
    print("post sweeps", n, ", nu = ", post[n-1])
    for j in range(post[n-1]):
        _, u0 = sweep(u0, pars, phi)

    return u0    

#test multigrid algorithm

### THERE IS STILL A SEGFAULT IN MULTIGRID

N = 64
beta = 1.
a = 1
delta = 2.
phi = np.zeros(N+1)

pars = (N, beta, a, delta)

n = 3
pre = [4, 2, 1]
post = [4, 2, 1]

gamma = 1
Nmeas = 100 #number of measurements

u = np.zeros(N+1)
u = multigrid(pre, post, n, gamma, pars, u, phi)
