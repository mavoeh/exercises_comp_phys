'''
In this code we calculate the magnetization deterministically by creating all possible spin configurations (there 
are 2^N possibilities for N sites and 2 possible spin directions) and summming over all of them. 
This leads to an almost exact result (very close to analytical result), but it takes some time, especially when
calculating for h in steps of 0.01.
'''

import numpy as np

def simulate(J, h, N, show_results = True):
    """
    calculates the partition function Z (numerical and analytical) as well
    as the average magnetization per spin for a certain combination of:
    
    interaction strength     J
    external field coupling  h
    number of spins          N
    
    diplays the results in the console if show_results = True
    
    
    returns: tuple of length 6
    
    external field coupling      h
    number of spins              N
    partition func (num)         Z
    partition func (analytical)  Z_analytical
    relative deviation           delta = |Z-Z_analytical|/Z
    average magnetization        m
    """
    # calculate the analytical result for the partition function
    Z_analytical = (np.exp(J)*(np.cosh(h) + np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N
    Z_analytical += (np.exp(J)*(np.cosh(h) - np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N
    
    # initialize 2d array for all possible spin configurations
    configs = np.zeros((2**N, N))

    # array containing 2**n for n from 0 to N-1
    n = np.linspace(0, N-1, N, dtype = np.int)
    n = 2**n
    
    """
    now fill up the configs array by alternating the sign of the
    spin for the individual spin sites at the periods contained
    in n to reach all possible combinations.
    
    Example for N = 3 spins configuration:
    
     4  2  1   Period at which the sign alternates
    _________
     1  1  1
     1  1 -1
     1 -1  1
     1 -1 -1
    -1  1  1
    -1  1 -1
    -1 -1  1
    -1 -1 -1
    
    """
    for i in range(N):
        configs[:,-i-1] = np.array( ([1]*n[i] + [-1]*n[i]) * n[-i-1])


    # define function that calculates the total energy (hamiltonian)
    # for a fixed J, h, and spin configuration s (1d numpy array of length N)
    def H(J, h, s):
        interact = np.sum(s[:-1]*s[1:]) + s[0]*s[-1]
        if N == 2:
            interact /= 2
        interact *= -J
        external = -h*np.sum(s)
        return interact + external
    
    # initialize and assign the boltzmann factors of the configs to z_array
    z_array = np.zeros(2**N)
    for i in range(len(z_array)):
        z_array[i] = np.exp(-H(J,h,configs[i]))
    
    # calculate and display the partition function (numerical and analytical)
    # as well as the absolute and relative deviations
    Z = np.sum(z_array)
    deltaZ = Z-Z_analytical
    delta = np.abs(deltaZ)/Z
    if show_results:
        print("partition function Z = {0:.20e}".format(Z))
        print("analytical result  Z = {0:.20e}".format(Z_analytical))

        print("\nresulting error: deltaZ = {0:e}".format(deltaZ))
        print("relative error    delta = {0:e}".format(delta))


    # from the boltzmann factors and partition func calculate probabilities
    P = z_array/Z
    
    # by multiplying the probability of a certain spin config with the spin value
    # for all spin sites and configs, the average magnetization is determined
    m = np.dot(P, configs)
    
    # average over all spin sites (could also just take one of the values because
    # they are all equal, as the problem is invariant under rotation of spin sites)
    m = np.sum(m)/len(m)
    

    if show_results: # display m
        print("\naverage magnetization <m> = {0:.3f}".format(m))
    
    return h, N, Z, Z_analytical, delta, m



J = 1 # initialize spin coupling to J = 1

# create array for values of h from -1, 1 in steps of 0.01
h_array = np.linspace(-1,1,201)
# same for N from 2 to 20 in interger steps
N_array = np.linspace(2, 20, 19, dtype = np.int)

# loop ove rall combinations
for N in N_array:
    for h in h_array:
        print("N = {0:d} from 20, h = {1:5.2f} from -1 to 1 in steps of 0.01".format(N,h), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_det.txt", "a")
        f.write("{0:.2f}\t{1:d}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\n".format(*simulate(J, h, N, show_results = False)))
        f.close()
