import numpy as np


def H(J, h, Nx, Ny, config):
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
    H0 = H(J, h, Nx, Ny, config)

    #sweep through the lattice
    for i in range(Nx):
        for j in range(Ny):

            #flip spin
            config[i,j] *= -1

            #calculate change in energy
            Hflip = H(J, h, Nx, Ny, config)
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


Nx = 3
Ny = 3

J = 1
h = -1

config = np.zeros((Nx, Ny))

for j in range(Nx):
    config[j] = np.random.choice([-1, 1], Ny) 

new = sweep(config, Nx, Ny, J, h)

