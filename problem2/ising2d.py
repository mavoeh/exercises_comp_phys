import numpy as np
import scipy.special

Jc = 0.440686793509772


def H(J, h, Nx, Ny, config):
        # calculates Hamiltonian for a Nx*Ny spin configuration

        #interaction term with four neighbours (in 2 dimensions)
        interact = np.sum(config[1:,:]*config[:-1,:])+np.sum(config[:, 1:]*config[:, :-1])
        if Nx > 2:
            interact += np.sum(config[0,:] *config[-1,:])
        if Ny > 2:
            interact += np.sum(config[:,0] *config[:,-1])
        interact *= -J
        #coupling to external magnetic field
        external = -h*np.sum(config)
        return interact + external

def sweep(config, Nx, Ny, J, h):

    '''
    Implements Metropolis-Hastings method for a spin configuration config in a lattice with Nx*Ny spins

    Input:
    random spin configuration   config    number of spins in x-dir    Nx
    number of spins in y-dir    Ny
    coupling constant           J
    magnetic coupling           h

    Output:
    configuration with smallest possible energy    listconfigs

    '''

    #sweep through the lattice
    for i in range(Nx):
        for j in range(Ny):

            #flip spin
            config[i,j] *= -1

            #calculate change in energy
            delta = -2*config[i,j]*(h + J*(config[(i-1)%Nx, j] + config[i, (j-1)%Ny] + config[i, (j+1)%Ny] + config[(i+1)%Nx,j]))

            #metropolis-hastings step
            y = np.random.uniform(0,1)

            if(y > np.exp(-delta)):
                config[i,j] *= -1

    return config

def simulate(J, h, Nx, Ny, Nmeas):
    
    '''
    Implements Metropolis-Hastings method for a spin configuration config in a lattice with Nx*Ny spins

    Input
    number of spins in x-dir    Nx
    number of spins in y-dir    Ny
    coupling constant           J
    magnetic coupling           h
    number of spin config       n
    number of measurements      Nmeas

    Output:
    J, h, Nx, Ny
    average multiplicaton       m
    statistical error           delm
    average energy              eps
    statistical error           deleps
    exact value for |m|         absm_exact
    exact value for eps         eps_exact        
    
    '''

    mray = np.zeros(Nmeas)
    epsray = np.zeros(Nmeas)
    

    for t in range(Nmeas):

        # sample spin configuration 
    
        config = np.random.choice([-1, 1], (Nx, Ny)) 

        config = sweep(config, Nx, Ny, J, h)

        #calculate m for each spin configuration 
        mray[t] = 1./Nx/Ny*np.sum(config)

        #calculate energy per site
        epsray[t] = 1/Nx/Ny*np.sum(H(J, h, Nx, Ny, config))


    #Calculate mean value of m and eps (average over all "measurements")
    m = np.mean(mray)
    absm = np.mean(np.abs(mray))
    eps = np.mean(epsray)

    #Calculate statistical error of m and Z
    delm = np.std(mray)
    delabsm = np.std(np.abs(mray))
    deleps = np.std(epsray)

    absm_exact = 0
    eps_exact = 0
    # exact magnetization per site and energy per site for h = 0
    if(h == 0):
        if(J > Jc):
            absm_exact = (1.-1./(np.sinh(2*J)**4))**(1./8.)

        eps_exact = -J/np.tanh(2*J)*(1 + 2./np.pi*(2*np.tanh(2*J)**2 - 1)*scipy.special.ellipk(4/np.cosh(2*J)**2*np.tanh(2*J)**2))

    return h, J, Nx, Ny, m, delm, eps, deleps, absm, delabsm, absm_exact, eps_exact


Nmeas = 100
N_array = np.arange(4, 21, 1)

#calculate m as a function of h for fixed J < 1, Nx = Ny between 4 and 20
# loop over all combinations
h_array = np.linspace(-1, 1, 21) # h in steps of 0.1
J = 0.5

for i in range(len(N_array)):
    N = N_array[i]
    for h in h_array:
        print("N = {0:d} from 20, h = {1:5.2f} from -1 to 1 in steps of 0.1".format(N,h), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_m.txt", "a")
        f.write("{0:.2f}\t{1:.2f}\t{2:d}\t{3:d}\t{4:e}\t{5:e}\t{6:e}\t{7:e}\t{8:e}\t{9:e}\t{10:e}\t{11:e}\n".format(*simulate(J, h, N, N, Nmeas)))
        f.close()

# calculate absm and eps for h=0 as a function of J in [.25, 1] and [.25, 2]
# for absm take abs value of m when plotting

J_array = np.linspace(0.25, 2, 100)
h = 0

for i in range(len(N_array)):
    N = N_array[i]
    for J in J_array:
        print("N = {0:d} from 20, J = {1:5.2f} from 0.25 to 2 in steps of ?".format(N,J), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_absm_eps.txt", "a")
        f.write("{0:.2f}\t{1:.2f}\t{2:d}\t{3:d}\t{4:e}\t{5:e}\t{6:e}\t{7:e}\t{8:e}\t{9:e}\t{10:e}\t{11:e}\n".format(*simulate(J, h, N, N, Nmeas)))
        f.close()

