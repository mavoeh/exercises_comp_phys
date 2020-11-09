import numpy as np
import matplotlib.pyplot as plt

font = {"fontname":"Times New Roman", "fontsize":18}

#load data into arrays
h, N, Z, delZ, m, delm = np.loadtxt("results_h100.txt", unpack = True)

# initialize empty array to store the data
data = np.zeros((19 , 201, 4))

# loop over all N
for n in range(19):
    mask = [N == n+2]
    # store the data in 3d array such that
    # 1st index = N - 2
    # 2nd index -> h
    # 3rd index -> variable: 0 = Z, 1 = delZ, 2 = m, 3 = delm
    data[n,:,:] = np.column_stack([Z[mask], delZ[mask], m[mask], delm[mask]])

def hplot(i, N = 20):
    # plots variable i against h (-1 to 1) for a certain N
    h = np.linspace(-1, 1, 201)
    plt.plot(h, data[N-2,:,i], label = "N = {}".format(N))

def Nplot(i, h):
    # plots variable i against N for certain h
    N = np.linspace(2, 20, 19)
    h_index = int(100*h + 100)
    plt.plot(N, data[:,h_index,i], label= "h = {}".format(h))
    


fig_hplot = plt.figure(figsize=(8,6))
for n in range(7):
    hplot(2, N=3*n+2)
    #plt.label("N = %s", 3*n+2)
plt.xlabel("external coupling $h$", **font)
plt.ylabel("average magnetization $<m>$", **font)
plt.grid(True)
plt.legend()
plt.savefig("hplot_h100.pdf", bosinches = "tight")


fig_Nplot = plt.figure(figsize=(8,6))
for n in range(21):
	Nplot(2, -1.+0.1*n)
  #plt.label("h = %s", -1+0.3*n)
plt.xlabel("number of spins $N$", **font)
plt.ylabel("average magnetization $<m>$", **font)
plt.grid(True)
plt.legend()
plt.savefig("Nplot_h100.pdf", bosinches = "tight")



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
plt.savefig("2d_h100.pdf", bosinches = "tight")



plt.show()
