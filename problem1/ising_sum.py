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
    partition func (analytical)  Z_analytical
    relative deviation           delta = |Z-Z_analytical|/Z
    average magnetization        m
    """
    # calculate the analytical result for the partition function
    Z_analytical = (np.exp(J)*(np.cosh(h) + np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N
    Z_analytical += (np.exp(J)*(np.cosh(h) - np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**N

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
    def zarray(J, h, n):
        z_array = np.zeros(n)
        for i in range(len(z_array)):
            z_array[i] = np.exp(-H(J,h,configs[i]))
        return z_array
    
    # calculate and display the partition function (numerical and analytical)
    # as well as the absolute and relative deviations

    def Z(z_array):
        return np.sum(zarray(J, h, n))

    Nmeas = 100
    mray = np.zeros(Nmeas)
    Zray = np.zeros(Nmeas)
    
    
    for t in range(Nmeas):

        # initialize 2d array for Nmeas random spin configurations
        configs = np.zeros((n, N))

        for j in range(n):
            configs[j] = np.random.choice([-1, 1], N) 

        
        #calculate m for each spin configuration
        m_array = np.zeros(n)
        for i in range(len(m_array)):
            m_array[i] = np.sum(configs[i])*np.exp(-H(J, h, configs[i]))


        #sum over all configs to get estimate for m and Z
        Zray[t] = Z(zarray(J, h, n))
        mray[t] = 1./N/Zray[t]*np.sum(m_array)

        #mray[t] = 1./N/20*(np.log(Z(zarray(J, h+1./20, n))) - np.log(Z(zarray(J, h-1./20, n))))


    #Calculate mean value of m and Z (average over all "measurements")

    Zres = np.sum(Zray)/Nmeas
    m = np.sum(mray)/Nmeas

    delZres = np.sqrt(np.sum((Zray-Zres)**2))
    delm = np.sqrt(np.sum((m-mray)**2))

    deltaZ = Zres-Z_analytical
    delta = np.abs(deltaZ)/Zres

    if show_results:
        print("partition function Z = {0:.20e}".format(Z))
        print("analytical result  Z = {0:.20e}".format(Z_analytical))

        print("\nresulting error: deltaZ = {0:e}".format(deltaZ))
        print("relative error    delta = {0:e}".format(delta))


    if show_results: # display m
        print("\naverage magnetization <m> = {0:.3f}".format(m))

    m_analytical = (np.exp(J)*(np.cosh(h) + np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**(N-1)*np.exp(J)*(np.sinh(h) + np.sinh(h)*np.cosh(h)/np.sqrt(np.sinh(h)**2 + np.exp(-4*J) ))
    m_analytical += (np.exp(J)*(np.cosh(h) - np.sqrt( np.sinh(h)**2 + np.exp(-4*J) )))**(N-1)*np.exp(J)*(np.sinh(h) - np.sinh(h)*np.cosh(h)/np.sqrt(np.sinh(h)**2 + np.exp(-4*J) ))
    m_analytical *= -1./Z_analytical


    return h, N, Zres, delZres, m, delm



J = 1 # initialize spin coupling to J = 1
no = np.repeat(100, 19)

# create array for values of h from -1, 1 in steps of 0.01
h_array = np.linspace(-1,1,201)
# same for N from 2 to 20 in interger steps
N_array = np.linspace(2, 20, 19, dtype = np.int)

# loop ove rall combinations
for i in range(len(N_array)):
    n = no[i]
    N = N_array[i]
    for h in h_array:
        print("N = {0:d} from 20, h = {1:5.2f} from -1 to 1 in steps of 0.01".format(N,h), end = "\r")
        # write output of the simulate function to a text file
        f = open("results_random.txt", "a")
        f.write("{0:.2f}\t{1:d}\t{2:e}\t{3:e}\t{4:e}\t{5:e}\n".format(*simulate(J, h, N, n, show_results = False)))
        f.close()

