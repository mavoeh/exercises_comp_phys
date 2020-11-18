import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def H(J, h, s):
    # calculates Hamiltonian for a Nx*Ny spin configuration

    #interaction term with four neighbours (in 2 dimensions)
    interact = np.sum(s[1:,:]*s[:-1,:]) + np.sum(s[:,1:]*s[:,:-1])
    interact += np.sum(s[0,:]*s[-1,:]) + np.sum(s[:,0]*s[:,-1])
    
    #coupling to external magnetic field
    external = -h*np.sum(s)
    return external - J*interact

@jit(nopython=True)
def deltaS(J, h, s, x, y, Nx, Ny): # calculates the change in action for a spinflip at site (x,y)
    return 2*s[x,y]*( J*( s[(x-1)%Nx,y] + s[x,(y-1)%Ny] + s[x,(y+1)%Ny] + s[(x+1)%Nx,y] ) + h )

#@jit(nopython=True)
def bootstrap_error(array, nBS):
    # performs a bootstrap error estimation for an array, generating nBS bootstrap samples  
    n = len(array)
    bsmean = np.zeros(nBS)
    for i in range(nBS):
        indices = np.random.randint(n,size=n) # random bootstrap indices
        bsmean[i] = array[indices].mean()
    return bsmean.std()

@jit(nopython=True)
def sweep(s, Nx, Ny, J, h):
    '''
    performs a metropolis hastings sweep through a 2d lattice site of size (Nx, Ny)
    with coupling constant J and external coupling h
    
    returns: spin config after sweep    s
             acceptance rate            counter/(Nx*Ny)
    '''
    # sweep through the lattice
    counter = 0
    for i in range(Nx*Ny):
        x = np.random.randint(0,Nx)
        y = np.random.randint(0,Ny)
        # metropolis hastings step
        if np.random.uniform(0,1) <= np.exp(-deltaS(J,h,s,x,y,Nx,Ny)):
            s[x,y] *= -1
            counter += 1 #accept and increment counter by 1
    return s, counter/(Nx*Ny)


def simulate(J, h, Nx, Ny, Nmeas, Ntherm, nBS):
    
    '''
    simulates a 2d-(Nx,Ny)-lattice with coupling constant J and external coupling h
    by generating a random lattice. First thermalize it by performing Nmeas sweeps
    without storing any values. Then perform Nmeas sweeps and store average magnetization,
    absolute value of average mag., total energy after each sweep

    returns:    
    m               average magnetization
    delm            bootstrap error of m
    mabs            average absolut magnetization
    dmabs           bootstraperror of mabs
    eps             energy per spin site
    deleps          bootstrap error of eps
    mabs_exact      analytical solution for mabs (only h=0)
    eps_exact       analytical solution for eps (only h= 0)
    accept_rate     acceptance rate of metropolis hastings step
    
    '''
    # initialize random array
    s = np.random.choice([-1,1], (Nx,Ny))
    mhist = np.zeros(Ntherm+Nmeas+1)
    mhist[0] = s.mean()
    for n in range(Ntherm):
        s, _ = sweep(s, Nx, Ny, J, h)
        mhist[n+1] = s.mean()

    # initialize array to store measurement values
    m_array = np.zeros(Nmeas)    # average magnetization
    mabs_array = np.zeros(Nmeas) # absolute value of average magnetization
    E_array = np.zeros(Nmeas)    # total energy of spin configuration
    p_array = np.zeros(Nmeas)    # acceptance rate
        
    for n in range(Nmeas):
        # perform the sweep and store spin config in s
        s, p = sweep(s, Nx, Ny, J, h)
        mhist[n+Ntherm+1] = s.mean()
        
        # store average magnetization in m_array
        m_array[n] = s.mean()
        mabs_array[n] = np.abs(m_array[n]).mean()
        
        # store energy in E_array
        E_array[n] = H(J, h, s)
        
        # store acceptance rate in p_array
        p_array[n] = p
    
    m = m_array.mean()
    delm = bootstrap_error(m_array, 100)
    
    mabs = np.abs(mabs_array).mean()
    dmabs = bootstrap_error(mabs_array, 100)
    
    eps = E_array.mean()/(Nx*Ny)
    deleps = bootstrap_error(E_array, 100)/(Nx*Ny)
    
    accept_rate = p_array.mean()

    mabs_exact = 0
    eps_exact = 0
    # exact magnetization per site and energy per site for h = 0
    if(h == 0):
        if(J > Jc):
            mabs_exact = (1.-1./(np.sinh(2*J)**4))**(1./8.)
        eps_exact = -J/np.tanh(2*J)*(1 + 2./np.pi*(2*np.tanh(2*J)**2 - 1)*scipy.special.ellipk(4/np.cosh(2*J)**2*np.tanh(2*J)**2))
        if np.abs(mabs-mabs_exact) > 0.1:
            plt.figure()
            plt.scatter(np.linspace(1,Ntherm+Nmeas+2,Ntherm+Nmeas+1),mhist)
            plt.title("Nx = {0:d}, Ny = {1:d}, J = {2:.2f}, h = {3:.2f}".format(Nx, Ny, J, h))
    return m, delm, mabs, dmabs, eps, deleps, mabs_exact, eps_exact, accept_rate


font = {"fontname":"Times New Roman", "fontsize":18}

def mplot(J, N_values, h_array):
    """
    plots m for 4 values of N contained in the list N_values
    vs the values of h cointained in h_array
    uses coupling contant J
    
    returns:
    singlefig   pyplot figure in which all plots are overlaid
    multifig    pyplot figure with 4 subplots
    """
    steps = len(h_array)
    data = np.zeros((steps, 9))
    
    multifig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,6))
    multifig.text(0.54, 0.02, "h", ha='center', **font)
    multifig.text(0.02, 0.525, r'$\langle m \rangle$', va='center', rotation='vertical', **font)
    
    singlefig, singleax = plt.subplots(1,1,figsize=(8,6))
    
    colors = ['magenta','lime','red','blue']
    for i, N in enumerate(N_values):
        for j, h in enumerate(h_array):
            
            data[j,:] = simulate (J, h, N, N, Nmeas, Ntherm, nBS)
        
        ax = axs[i//2,i%2]
        ax.errorbar(h_array, data[:,0], data[:,1],
                    label = ("N = %d"%N),
                    linestyle = "none",
                    capsize = 2,
                    capthick = 2)
        ax.legend(loc=0)
        ax.grid(True)
        singleax.scatter(h_array, data[:,0],
                         s = 30,
                         edgecolor="none",
                         c = colors[i],
                         label = ("N = %d"%N))
        singleax.set_xlim(min(h_array)-0.02,max(h_array)+0.02)
        singleax.set_ylim(min(data[:,0])-0.02,max(data[:,0])+0.02)
        singleax.set_xlabel("h", **font)
        singleax.set_ylabel(r'$\langle m \rangle$', **font)
        singleax.grid(True)
        singleax.legend(loc=2)
    multifig.tight_layout(rect=[0.04, 0.04, 1, 1])
    return singlefig, multifig
    
def mabsplot(N_values, J_array):
    """
    plots absolute value of m for 4 values of N contained in the list N_values
    vs the values of J cointained in J_array
    uses external coupling h = 0
    
    returns:
    singlefig   pyplot figure in which all plots are overlaid
    multifig    pyplot figure with 4 subplots
    """
    steps = len(J_array)
    data = np.zeros((steps, 9))
    h = 0
    
    multifig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,6))
    multifig.text(0.535, 0.02, "J", ha='center', **font)
    multifig.text(0.02, 0.525, r'$\langle m \rangle$', va='center', rotation='vertical', **font)
    
    singlefig, singleax = plt.subplots(1,1,figsize=(8,6))
    
    colors = ['magenta','lime','red','blue']
    for i, N in enumerate(N_values):
        for j, J in enumerate(J_array):
            
            data[j,:] = simulate (J, h, N, N, Nmeas, Ntherm, nBS)
        
        ax = axs[i//2,i%2]
        ax.errorbar(J_array, data[:,2], data[:,3],
                    label = ("N = %d"%N),
                    linestyle = "none",
                    capsize = 2,
                    capthick = 2)
        ax.legend(loc=4)
        ax.grid(True)
        singleax.scatter(J_array, data[:,2],
                         s = 30,
                         edgecolor="none",
                         c = colors[i],
                         label = ("N = %d"%N))
        singleax.set_xlim(min(J_array)-0.02,max(J_array)+0.02)
        singleax.set_ylim(min(data[:,2])-0.02,max(data[:,2])+0.02)
        singleax.set_xlabel("h", **font)
        singleax.set_ylabel(r'$\langle m \rangle$', **font)
        singleax.grid(True)
        singleax.legend(loc=4)
    multifig.tight_layout(rect=[0.04, 0.04, 1, 1])
    return singlefig, multifig

Jc = 0.440686793509772
Nmeas = 200
Ntherm = 100
nBS = 100
N = [5,10,15,20]   
        
#m_singlefig, m_multifig = mplot(0.3, N, np.linspace(-1,1,21))
#m_multifig.savefig("m_multifig.pdf")
#m_singlefig.savefig("m_singlefig.pdf")

mabs_singlefig, mabs_multifig = mabsplot(N, np.linspace(0.25,2,21))
mabs_multifig.savefig("test1.pdf")
mabs_singlefig.savefig("test2.pdf")



plt.show()




