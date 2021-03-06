import numpy as np
import matplotlib.pyplot as plt

font = {"fontname":"Times New Roman", "fontsize":18}

#load data from .txt documents obtained from ising2d.py into arrays: here for m(h) calculation
h, J, Nx, Ny, m, delm, eps, deleps, absm, delabsm, absm_exact, eps_exact = np.loadtxt("results_m.txt", unpack = True)

#depending on which step for h was calculated use 
h10 = 21 #for h in steps of 0.1
h100 = 201 #for h in steps of 0.01

# initialize empty array to store the data
data = np.zeros((17 , h10, 8))

# loop over all N
for n in range(17):
    mask = [Nx == n+4]
    # store the data in 3d array such that
    # 1st index = N - 2
    # 2nd index -> h
    # 3rd index -> variable: 0 = m, 1 = delm, 2 = eps, 3 = deleps, 4 = absm_exact, 5 = eps_exact
    data[n,:,:] = np.column_stack([m[mask], delm[mask], eps[mask], deleps[mask], absm[mask], delabsm[mask], absm_exact[mask], eps_exact[mask]])

def hplot(i, N = 20):
    # plots variable i against h (-1 to 1) for a certain N
    h = np.linspace(-1, 1, h10)
    plt.plot(h, data[N-4,:,i], label = "N = {}".format(N))

#Plot m depending on h for a few different N
fig_hplot = plt.figure(figsize=(8,6))
harray = np.linspace(-1, 1, h10)
for n in range(17):
    hplot(0, N=n+4)
plt.xlabel("external coupling $h$", **font)
plt.ylabel(r"average magnetization $\langle m\rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("m_h10.pdf", bosinches = "tight")
plt.show()



#load data from .txt documents obtained from ising2d.py into arrays: here for absm and eps calculation
h, J, Nx, Ny, m, delm, eps, deleps, absm, delabsm, absm_exact, eps_exact = np.loadtxt("results_absm_eps.txt", unpack = True)

# initialize empty array to store the data
data = np.zeros((17 , 100, 8))

# loop over all N
for n in range(17):
    mask = [Nx == n+4]
    # store the data in 3d array such that
    # 1st index = N - 2
    # 2nd index -> J
    # 3rd index -> variable: 0 = m, 1 = delm, 2 = eps, 3 = deleps, 4=absm, 5 = delabsm, 6 = absm_exact, 7 = eps_exact
    data[n,:,:] = np.column_stack([m[mask], delm[mask], eps[mask], deleps[mask], absm[mask], delabsm[mask], absm_exact[mask], eps_exact[mask]])

def Jplot_m(i, N = 20):
    # plots variable absm against J (-0.25 to 1) for a certain N
    absm = data[N-4,:,i]
    J = np.linspace(0.25, 2, len(absm))
    mask = J <= 1
    plt.plot(J[mask]**(-1), absm[mask], label = "N = {}".format(N))


#Plot |m| depending on J for a few different N
fig_Jplot = plt.figure(figsize=(8,6))
for n in range(17):
    Jplot_m(4, N=n+4)
Jarray = np.linspace(0.25, 1, len(data[4,:,6]))
plt.plot(1./Jarray, data[4,:,6], label = "exact")
plt.xlabel("coupling constant $J^{-1}$", **font)
plt.ylabel(r"average value of absolute magnetization $\langle | m |\rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("absm_J100.pdf", bosinches = "tight")

#Plot m depending on J for a few different N
fig_Jplot = plt.figure(figsize=(8,6))
for n in range(17):
    Jplot_m(0, N=n+4)
Jarray = np.linspace(0.25, 1, len(data[4,:,6]))
plt.plot(1./Jarray, data[4,:,6], label = "exact")
plt.xlabel("coupling constant $J^{-1}$", **font)
plt.ylabel(r"average value of magnetization $\langle m \rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("m_J100.pdf", bosinches = "tight")


def Jplot_eps(i, N = 20):
    # plots variable i against J for a certain N
    J = np.linspace(0.25, 2, 100)
    plt.plot(J, data[N-4,:,i], label = "N = {}".format(N)) #eps data times two?

#Plot eps depending on J for a few different N
fig_Jplot = plt.figure(figsize=(8,6))
for n in range(17):
    Jplot_eps(2, N=n+4)
Jarray = np.linspace(0.25, 2, len(data[10,:,7]))
plt.plot(Jarray, data[8,:,7], label = "exact")
plt.xlabel("coupling constant $J$", **font)
plt.ylabel(r"average energy per site $\langle \epsilon \rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("eps_J100.pdf", bosinches = "tight")
plt.show()



