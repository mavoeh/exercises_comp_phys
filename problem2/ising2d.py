import numpy as np
import scipy.special

Jc = 0.440686793509772


def H(J, h, config):
        # calculates Hamiltonian for a Nx*Ny spin configuration

        #interaction term with four neighbours (in 2 dimensions)
        interact = np.sum(config[:-1,:-1]*config[:-1,1:]*config[1:,:-1])
        interact += np.sum(config[0]*config[-1]*np.roll(config[-1], -1)) + np.sum(np.transpose(config)[-1]*np.transpose(config)[0]*np.roll(np.transpose(config)[-1], -1))
        interact -= config[-1,-1]*config[-1,0]*config[0,-1]
        interact *= -J

        #coupling to external magnetic field
        external = -h*np.sum(config)
        return interact + external

def sweep(config, Nx, Ny, J, h):

    '''
    Implements Metropolis-Hastings method for a spin configuration config in a lattice with Nx*Ny spins

    Input:
    random spin configuration   config
    number of spins in x-dir    Nx
    number of spins in y-dir    Ny
    coupling constant           J
    magnetic coupling           h

    Output:
    configuration with smallest possible energy    config

    '''
    H0 = H(J, h, config)

    #sweep through the lattice
    for i in range(Nx):
        for j in range(Ny):

            #flip spin
            config[i,j] *= -1

            #calculate change in energy
            Hflip = H(J, h, config)
            delta = Hflip - H0

            #metropolis-hastings step (how to choose step?)
            y = np.random.choice(np.arange(0, 1, 0.0001))

            if(y < np.exp(-delta)):
                H0 = Hflip
                #print("flip:", y, np.exp(-delta))
            else:
                config[i,j] *= -1
                #print("noflip:", y, np.exp(-delta))
    return config



def simulate(J, h, Nx, Ny, n, Nmeas):
    
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
    
    '''

    # initialize and assign the boltzmann factors of the configs to z_array
    def zarray(J, h, n, listconfigs):
        z_array = np.zeros(n)
        for i in range(len(z_array)):
            z_array[i] = np.exp(-H(J,h, listconfigs[i]))
        return z_array
    
    # calculate the partition function

    def Z(z_array):
        return np.sum(z_array)

    mray = np.zeros(Nmeas)
    Zray = np.zeros(Nmeas)
    epsray = np.zeros(Nmeas)
    

    for t in range(Nmeas):

        # initialize list of 2d array for n random spin configurations
        listconfigs = []

        for i in range(n):
                config = np.zeros((Nx, Ny))

                for j in range(Nx):
                    config[j] = np.random.choice([-1, 1], Ny) 

                listconfigs.append(config)

        #for each spin configuration in listconfigs do a sweep through the lattice

        for config in listconfigs:
            sweep(config, Nx, Ny, J, h)

        
        #calculate m for each spin configuration (here: by using formula (4) on the exercise sheet)
        m_array = np.zeros(n)
        for i in range(len(m_array)):
            m_array[i] = np.sum(listconfigs[i])*np.exp(-H(J, h, listconfigs[i]))

        #calculate energy per site
        eps_array = np.zero(n)
        for i in range(n):
            eps_array[i] = H(J, h, lisconfigs[i]) 


        #sum over all configs to get estimate for m, eps and Z
        epsray[t] = 1/Nx/Ny*np.sum(eps_array)
        Zray[t] = Z(zarray(J, h, n, listconfigs))
        mray[t] = 1./N/Zray[t]*np.sum(m_array)


    #Calculate mean value of m and Z (average over all "measurements")

    Zres = np.mean(Zray)
    m = np.mean(mray)
    eps = np.mean(epsray)

    #Calculate statistical error of m and Z and relative deviation to analytical result. Display results if show_results = True
    delZres = np.std(Zray)
    delm = np.std(mray)
    deleps = np.mean(epsray)


    absm_exact = 0
    eps_exact = 0
    # exact magnetization per site and energy per site for h = 0
    if(h == 0):
        if(J > Jc):
            absm_exact = (1.-1./(np.sinh(2*J)**4))**(1./8.)

        eps_exact = -J/np.tanh(2*J)*(1 + 2./np.pi*(2*np.tanh(2*J)**2 - 1)*scipy.special.ellipk(4/np.cosh(2*J)**2*np.tanh(2*J)**2))

    return h, J, Nx, Ny, m, delm, eps, deleps, absm_exact, eps_exact


n = 10
Nmeas = 100
N_array = np.arange(4, 21, 1)

#calculate m as a function of h for fixed J < 1, Nx = Ny between 4 and 20
# loop ove rall combinations
h_array = np.linspace(-1, 1, 21) # h in steps of 0.1
J = 0.5

for i in range(len(N_array)):
    N = N_array[i]
    for h in h_array:
        print("N = {0:d} from 20, h = {1:5.2f} from -1 to 1 in steps of 0.1".format(N,h), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_m.txt", "a")
        f.write("{0:.2f}\t{1:.2f}\t{2:d}\t{3:d}\t{4:e}\t{5:e}\t{6:e}\t{7:e}\t{8:e}\t{9:e}\n".format(*simulate(J, h, N, N, n, Nmeas)))
        f.close()

# calculate absm and eps for h=0 as a function of J in [.25, 1] and [.25, 2]
# for absm take abs value of m when plotting :)

J_array = np.linspace(0.25, 2, 100)
h = 0

for i in range(len(N_array)):
    N = N_array[i]
    for J in J_array:
        print("N = {0:d} from 20, J = {1:5.2f} from 0.25 to 2 in steps of ?".format(N,J), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_absm_eps.txt", "a")
        f.write("{0:.2f}\t{1:.2f}\t{2:d}\t{3:d}\t{4:e}\t{5:e}\t{6:e}\t{7:e}\t{8:e}\t{9:e}\n".format(*simulate(J, h, N, N, n, Nmeas)))
        f.close()




