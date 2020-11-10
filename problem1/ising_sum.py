'''
In this code we implement n random spin configurations for which we calculate the partition function Z and the
average magnetization m. This we do Nmeas time (for Nmeas 'measurements') and average over all values for Z and m. 
In doing so, we hope to get a roughly accurate result for m and Z in (relatively) little computing time.
Have .txt tables of results for:
n = 100, Nmeas = 1000, h in steps of 0.1 (results_sum1000.txt)
n = 100, Nmeas = 100, h in steps of 0.1 (results_sum100.txt)
n = 100, Nmeas = 100, h in stepps of 0.01 (results_h100.txt)

'''

import numpy as np

def simulate(J, h, N, n, show_results = True):
    """
    calculates the partition function Z (numerical and analytical) as well
    as the average magnetization per spin for a certain combination of:
    
    interaction strength     J
    external field coupling  h
    number of spins          N
    number of configurations n
    
    diplays the results in the console if show_results = True
    
    
    returns: tuple of length 6
    
    external field coupling      h
    number of spins              N
    partition func (num)         Z
    statistical error for Z      delZres
    average magnetization        m
    statistical error for m      delm
    analytical result for m      m_analytical
    relative deviations          delta = |m-m_analytical|/m

    """
    # calculate the analytical result for the partition function and magnetization m
    Z_analytical = (np.exp(J)*(np.cosh(h) + np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N
    Z_analytical += (np.exp(J)*(np.cosh(h) - np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N


    m_analytical = (np.exp(J)*(np.cosh(h) + np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**(N-1)*np.exp(J)*(np.sinh(h) + np.sinh(h)*np.cosh(h)/np.sqrt(np.sinh(h)**2 + np.exp(-4*J) ))
    m_analytical += (np.exp(J)*(np.cosh(h) - np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**(N-1)*np.exp(J)*(np.sinh(h) - np.sinh(h)*np.cosh(h)/np.sqrt(np.sinh(h)**2 + np.exp(-4*J) ))
    m_analytical *= -1./Z_analytical


    #define function that calculates the total energy (hamiltonian)
    # for a fixed J, h, and spin configuration s (1d numpy array of length N)
    def H(J, h, s):
        interact = np.sum(s[:-1]*s[1:]) + s[0]*s[-1]
        if N == 2:
            interact /= 2
        interact *= -J
        external = -h*np.sum(s)
        return interact + external
    
    # initialize and assign the boltzmann factors of the configs to z_array
    def zarray(J, h, n):
        z_array = np.zeros(n)
        for i in range(len(z_array)):
            z_array[i] = np.exp(-H(J,h,configs[i]))
        return z_array
    
    # calculate the partition function

    def Z(z_array):
        return np.sum(zarray(J, h, n))

    # define the number of 'measurements' Nmes i.e, the number of times we calculate m for n spin configurations
    Nmeas = 1000
    mray = np.zeros(Nmeas)
    Zray = np.zeros(Nmeas)
    

    #calculate Z and m for each spin configuration and save values in array Zray and mray of length Nmeas 
    for t in range(Nmeas):

        # initialize 2d array for n random spin configurations
        configs = np.zeros((n, N))

        for j in range(n):
            configs[j] = np.random.choice([-1, 1], N) 

        
        #calculate m for each spin configuration (here: by using formula (4) on the exercise sheet)
        m_array = np.zeros(n)
        for i in range(len(m_array)):
            m_array[i] = np.sum(configs[i])*np.exp(-H(J, h, configs[i]))


        #sum over all configs to get estimate for m and Z
        Zray[t] = Z(zarray(J, h, n))
        mray[t] = 1./N/Zray[t]*np.sum(m_array)


    #Calculate mean value of m and Z (average over all "measurements")

    Zres = np.sum(Zray)/Nmeas
    m = np.sum(mray)/Nmeas

    #Calculate statistical error of m and Z and relative deviation to analytical result. Display results if show_results = True
    delZres = np.sqrt(np.sum((Zray-Zres)**2))
    delm = np.sqrt(np.sum((m-mray)**2))

    deltam = m-m_analytical
    delta = np.abs(deltam)/m

    if show_results:
        print("partition function Z = {0:.20e}".format(Z))
        print("analytical result  Z = {0:.20e}".format(Z_analytical))

        print("\nresulting error: deltaZ = {0:e}".format(deltaZ))
        print("relative error    delta = {0:e}".format(delta))


    if show_results: # display m
        print("\naverage magnetization <m> = {0:.3f}".format(m))

    return h, N, Zres, delZres, m, delm, m_analytical, delta



J = 1 # initialize spin coupling to J = 1
no = np.repeat(100, 19) #number of spin configurations calculated for each 'measurement'

# create array for values of h from -1, 1 in steps of 0.01 (h100) or 0.1 (h10)
h100 = 201
h10 = 21
h_array = np.linspace(-1,1, h10)
# same for N from 2 to 20 in integer steps
N_array = np.linspace(2, 20, 19, dtype = np.int)

# loop ove rall combinations
for i in range(len(N_array)):
    n = no[i]
    N = N_array[i]
    for h in h_array:
        print("N = {0:d} from 20, h = {1:5.2f} from -1 to 1 in steps of 0.01".format(N,h), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_random.txt", "a")
        f.write("{0:.2f}\t{1:d}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\t{6:e}\t{7:e}\n".format(*simulate(J, h, N, n, show_results = False)))
        f.close()

