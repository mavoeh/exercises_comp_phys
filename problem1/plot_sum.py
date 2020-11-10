'''
Plots average magnetization dependent of number of spin sites N for certain values of the external field 
couling h, and dependet on h for certain N. 
In this code we plot the results obtained by running the code 'ising_sum.py'.
'''

import numpy as np
import matplotlib.pyplot as plt

font = {"fontname":"Times New Roman", "fontsize":18}

#load data from .txt documents obtained from ising_sum.py into arrays
h, N, Z, delZ, m, delm, m_analytical, delta = np.loadtxt("results_sum1000.txt", unpack = True)

#depending on which step for h was calculated use 
h10 = 21 #for h in steps of 0.1
h100 = 201 #for h in steps of 0.01

# initialize empty array to store the data
data = np.zeros((19 , h10, 6))

# loop over all N
for n in range(19):
    mask = [N == n+2]
    # store the data in 3d array such that
    # 1st index = N - 2
    # 2nd index -> h
    # 3rd index -> variable: 0 = Z, 1 = delZ, 2 = m, 3 = delm, 4 = m_analytical, 5 = delta
    data[n,:,:] = np.column_stack([Z[mask], delZ[mask], m[mask], delm[mask], m_analytical[mask], delta[mask]])

def hplot(i, N = 20):
    # plots variable i against h (-1 to 1) for a certain N
    h = np.linspace(-1, 1, h10)
    plt.plot(h, data[N-2,:,i], label = "N = {}".format(N))

def Nplot(i, h):
    # plots variable i against N for certain h
    N = np.linspace(2, 20, 19)
    h_index = int(10*h + 10) #change factor = 1/hstep 
    plt.plot(N, data[:,h_index,i], label= "h = {0:.1f}".format(h)) 
    

#Plot m depending on h for a few different N
fig_hplot = plt.figure(figsize=(8,6))
harray = np.linspace(-1, 1, h10)
for n in range(5):
    hplot(2, N=4*n+2)
#plt.errorbar(harray, data[8,:,2], data[8,:,3], fmt = ".", color = "green")
plt.xlabel("external coupling $h$", **font)
plt.ylabel(r"average magnetization $\langle m\rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("hplot_sum1000.pdf", bosinches = "tight")

#Plot m depending on h for N = 10 with errors for m
fig_hsingle = plt.figure(figsize=(8,6))
hplot(2, N=10)
harray = np.linspace(-1, 1, h10)
plt.errorbar(harray, data[8,:,2], data[8,:,3], fmt = ".", color = "blue")
plt.xlabel("external coupling $h$", **font)
plt.ylabel(r"average magnetization $\langle m\rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("hsingle_sum1000.pdf", bosinches = "tight")

#plot m depending on N for a few different h, one with errors for m 
fig_Nplot = plt.figure(figsize=(8,6))
Narray = np.linspace(2, 20, 19)
for n in range(5):
  Nplot(2, -1.+0.5*n)
plt.errorbar(Narray, data[:,10,2], data[:,10,3], fmt = ".", color = "green")
plt.xlabel("number of spins $N$", **font)
plt.ylabel(r"average magnetization $\langle m\rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("Nplot_sum1000.pdf", bosinches = "tight")

#plot m in a 2d plot against h and N
fig = plt.figure(figsize=(8,6))
plt.imshow(data[:,:,2].T,
           aspect="auto",
           extent = [2,20,-1,1],
           origin = "lower",
           interpolation = "bilinear")
plt.colorbar()
plt.xticks(range(2,21,2))
plt.yticks(np.linspace(-1,1,5))
plt.xlabel("number of spins $N$", **font)
plt.ylabel("external coupling $h$", **font)
plt.savefig("2d_sum1000.pdf", bosinches = "tight")

delta_fig = plt.figure(figsize=(8,6))
plt.imshow(data[:,:,5].T,
           aspect="auto",
           extent = [2,20,-1,1],
           origin = "lower",
           interpolation = "bilinear")
plt.colorbar()
plt.xticks(range(2,21,2))
plt.yticks(np.linspace(-1,1,5))
plt.xlabel("number of spins $N$", **font)
plt.ylabel("external coupling $h$", **font)
plt.savefig("2d_delta_sum1000.pdf", bosinches = "tight")

plt.show()
