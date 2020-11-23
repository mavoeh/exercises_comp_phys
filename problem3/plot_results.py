import numpy as np
import matplotlib.pyplot as plt

font = {"fontname":"Times New Roman", "fontsize":18}


#load data from .txt documents obtained from ising2d.py into arrays: here for absm and eps calculation
N, J, m, delm, eps, deleps, acc, m_exact, eps_exact = np.loadtxt("results.txt", unpack = True)

# initialize empty array to store the data
data = np.zeros((16, 100, 7))

# loop over all N
for n in range(16):
    mask = [N == n+5]
    # store the data in 3d array such that
    # 1st index = N - 2
    # 2nd index -> J
    # 3rd index -> variable: 0 = m, 1 = delm, 2 = eps, 3 = deleps, 4=acc, 5=m_exact, 6 = eps_exact
    data[n,:,:] = np.column_stack([m[mask], delm[mask], eps[mask], deleps[mask], acc[mask], m_exact[mask], eps_exact[mask]])

def Jplot(i, j, N = 20):
    # plots variable i against J for a certain N
    J = np.linspace(0.25, 2, 100)
    plt.errorbar(J, data[N-5,:,i], data[N-5,:,j], label = "N = {}".format(N), fmt  = "x") #eps data times two?

#Plot m depending on J for a few different N
fig_Jplot = plt.figure(figsize=(8,6))
Jarray = np.linspace(0.2, 2, 100)
for n in range(4):
    Jplot(0, 1, N=n*4+5)
    plt.plot(Jarray, data[4,:,5], label = "exact for N = {}".format(n*4+5))
plt.xlabel("coupling constant $J$", **font)
plt.ylabel(r"average value of magnetization $\langle m \rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("m_J100.pdf")
plt.show()


#Plot eps depending on J for a few different N
fig_Jplot = plt.figure(figsize=(8,6))
Jarray = np.linspace(0.2, 2, 100)
for n in range(4):
    Jplot(2, 3, N=n*4+5)
    plt.plot(Jarray, data[4,:,6], label = "exact for N = {}".format(n*4+5))
plt.xlabel("coupling constant $J$", **font)
plt.ylabel(r"average energy per site $\langle \epsilon \rangle$", **font)
plt.grid(True)
plt.legend()
plt.savefig("eps_J100.pdf")
plt.show()



