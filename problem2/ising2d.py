import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from numba import jit

Jc = 0.440686793509772

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
def deltaS(J, h, s, x, y, Nx, Ny):
    return 2*s[x,y]*( J*( s[(x-1)%Nx,y] + s[x,(y-1)%Ny] + s[x,(y+1)%Ny] + s[(x+1)%Nx,y] ) + h )

@jit(nopython=True)
def sweep(s, Nx, Ny, J, h):
    '''
    Implements Metropolis-Hastings method for a spin configuration config in a lattice with Nx*Ny spins

    Input:
    random spin configuration   s
    number of spins in x-dir    Nx
    number of spins in y-dir    Ny
    coupling constant           J
    magnetic coupling           h

    Output:
    configuration after sweep    s
    '''
    # sweep through the lattice
    counter = 0
    for i in range(Nx):
        for j in range(Ny):
            # metropolis hastings step
            if np.random.uniform(0,1) <= np.exp(-deltaS(J,h,s,i,j,Nx,Ny)):
                s[i,j] *= -1
                counter += 1 #accept and increment counter by 1
    return s, counter/(Nx*Ny)


def simulate(J, h, Nx, Ny, Nmeas, Ntherm):
    
    '''
    Implements Metropolis-Hastings method for a spin configuration config in a lattice with Nx*Ny spins

    Input
    number of spins in x-dir    Nx
    number of spins in y-dir    Ny
    coupling constant           J
    magnetic coupling           h
    number of measurements      Nmeas

    Output:
    
    '''
    # initialize random array
    s = np.random.choice([-1,1], (Nx,Ny))
    for n in range(Ntherm):
        s, _ = sweep(s, Nx, Ny, J, h)
    """
    # thermalize (perform a few sweeps starting at random lattice site)
    # initialize random starting point
    for n in range(Ntherm*Nx*Ny):
        x = np.random.randint(0,1)
        y = np.random.randint(0,1)
        # metropolis hastings step
        if np.random.uniform(0,1) <= np.exp(-deltaS(J,h,s,x,y,Nx,Ny)):
            s[x,y] *= -1
    """
    # initialize array to store average magnetizations and total energy for each sweep
    m_array = np.zeros(Nmeas)
    mabs_array = np.zeros(Nmeas)
    E_array = np.zeros(Nmeas)
    
    p_array = np.zeros(Nmeas)
        
    for n in range(Nmeas):
        # perform the sweep and store spin config in s
        s, p = sweep(s, Nx, Ny, J, h)
        
        # store average magnetization in m_array
        m_array[n] = s.mean()
        mabs_arr[n] = np.abs(s).mean()
        
        # store energy in E_array
        E_array[n] = H(J, h, s)
        
        p_array[n] = p
    
    m = m_array.mean()
    delm = m_array.std()
    
    eps = E_array.mean()/(Nx*Ny)
    deleps = E_array.std()/(Nx*Ny)
    
    accept_rate = p_array.mean()
    daccept_rate = p_array.std()

    absm_exact = 0
    eps_exact = 0
    # exact magnetization per site and energy per site for h = 0
    if(h == 0):
        if(J > Jc):
            absm_exact = (1.-1./(np.sinh(2*J)**4))**(1./8.)

        eps_exact = -J/np.tanh(2*J)*(1 + 2./np.pi*(2*np.tanh(2*J)**2 - 1)*scipy.special.ellipk(4/np.cosh(2*J)**2*np.tanh(2*J)**2))
    if np.abs(eps - eps_exact) > 0.2:
        print(J)
        print(s)
    return m, delm, eps, deleps, absm_exact, eps_exact, accept_rate, daccept_rate


Nmeas = 1000
Nx = 10
Ny = 10
#h = 0
steps = 21

data = np.zeros((steps, 8))
"""
Jarray = np.linspace(0.25,2,steps)
for i, J in enumerate(Jarray):
    data[i,:] = simulate(J, h, Nx, Ny, Nmeas, 100)
    
plt.errorbar(Jarray, data[:,0], data[:,1],
             c="red",
             linestyle="none",
             capsize=3,
             capthick=3)

plt.errorbar(Jarray, data[:,6], data[:,7], c="cyan")

plt.plot(Jarray, data[:,5], c="orange")

plt.show()
"""
J = 0.3
harray = np.linspace(-0.5,0.5,steps)
for i, h in enumerate(harray):
    data[i,:] = simulate(J, h, Nx, Ny, Nmeas, 0)
    
plt.errorbar(harray, data[:,0], data[:,1],
             c="red",
             linestyle="none",
             capsize=3,
             capthick=3)

#plt.errorbar(Jarray, data[:,6], data[:,7], c="cyan")

#plt.plot(Jarray, data[:,5], c="orange")

plt.show() 



