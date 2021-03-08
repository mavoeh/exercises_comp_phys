import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman', size=12)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid'] = True

#first plot phi = 180, theta1 = theta2 = 39 and Elab = 13 MeV

S, sigma = np.loadtxt("txtfiles/phi180th39E13.txt", unpack = True, skiprows = 1)

plt.figure()
plt.plot(S, sigma)
plt.xlabel("$S$ (MeV)")
plt.ylabel("d$^3 \sigma$/d$\Omega_1$d$\Omega_2$d$S$ (mb sr$^{-2}$ MeV$^{-1}$)")
plt.tight_layout()
plt.savefig("plots/phi180th39E13.pdf")

#now plot fsi peaks for different binding energies phi = 0, Elab = 13 MeV

#theta = 20.5
S1, sig1 = np.loadtxt("txtfiles/phi0th205E13e125.txt", unpack = True, skiprows = 1)
pos1 = S1[0]
S1 = S1[1:]
sig1 = sig1[1:]
S2, sig2 = np.loadtxt("txtfiles/phi0th205E13e225.txt", unpack = True, skiprows = 1)
pos2 = S2[0]
S2 = S2[1:]
sig2 = sig2[1:]
S3, sig3 = np.loadtxt("txtfiles/phi0th205E13e325.txt", unpack = True, skiprows = 1)
pos3 = S3[0]
S3 = S3[1:]
sig3 = sig3[1:]

plt.figure()
plt.plot(S1, sig1, label = "$\epsilon_d = -2.125$ MeV", color = "firebrick")
plt.plot(S2, sig2, label = "$\epsilon_d = -2.225$ MeV", color = "navy")
plt.plot(S3, sig3, label = "$\epsilon_d = -2.325$ MeV", color = "forestgreen")
plt.plot(np.repeat(pos1,100), np.linspace(0, sig1.max(), 100), color = "firebrick")
plt.plot(np.repeat(pos2,100), np.linspace(0, sig2.max(), 100), color = "navy")
plt.plot(np.repeat(pos3,100), np.linspace(0, sig3.max(), 100), color = "forestgreen")
plt.xlabel("$S$ (MeV)", fontsize = 14)
plt.ylabel("d$^3 \sigma$/d$\Omega_1$d$\Omega_2$d$S$ (mb sr$^{-2}$ MeV$^{-1}$)", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.savefig("plots/phi0th205E13.pdf")


#theta = 28.0
S1, sig1 = np.loadtxt("txtfiles/phi0th28E13e125.txt", unpack = True, skiprows = 1)
pos1 = S1[0]
S1 = S1[1:]
sig1 = sig1[1:]
S2, sig2 = np.loadtxt("txtfiles/phi0th28E13e225.txt", unpack = True, skiprows = 1)
pos2 = S2[0]
S2 = S2[1:]
sig2 = sig2[1:]
S3, sig3 = np.loadtxt("txtfiles/phi0th28E13e325.txt", unpack = True, skiprows = 1)
pos3 = S3[0]
S3 = S3[1:]
sig3 = sig3[1:]

plt.figure()
plt.plot(S1, sig1, label = "$\epsilon_d = -2.125$ MeV", color = "firebrick")
plt.plot(S2, sig2, label = "$\epsilon_d = -2.225$ MeV", color = "navy")
plt.plot(S3, sig3, label = "$\epsilon_d = -2.325$ MeV", color = "forestgreen")
plt.plot(np.repeat(pos1,100), np.linspace(0, sig1.max(), 100), color = "firebrick")
plt.plot(np.repeat(pos2,100), np.linspace(0, sig2.max(), 100), color = "navy")
plt.plot(np.repeat(pos3,100), np.linspace(0, sig3.max(), 100), color = "forestgreen")
plt.xlabel("$S$ (MeV)", fontsize = 14)
plt.ylabel("d$^3 \sigma$/d$\Omega_1$d$\Omega_2$d$S$ (mb sr$^{-2}$ MeV$^{-1}$)", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.savefig("plots/phi0th28E13.pdf")

#theta = 35.5
S1, sig1 = np.loadtxt("txtfiles/phi0th355E13e125.txt", unpack = True, skiprows = 1)
pos1 = S1[0]
S1 = S1[1:]
sig1 = sig1[1:]
S2, sig2 = np.loadtxt("txtfiles/phi0th355E13e225.txt", unpack = True, skiprows = 1)
pos2 = S2[0]
S2 = S2[1:]
sig2 = sig2[1:]
S3, sig3 = np.loadtxt("txtfiles/phi0th355E13e325.txt", unpack = True, skiprows = 1)
pos3 = S3[0]
S3 = S3[1:]
sig3 = sig3[1:]

plt.figure()
plt.plot(S1, sig1, label = "$\epsilon_d = -2.125$ MeV", color = "firebrick")
plt.plot(S2, sig2, label = "$\epsilon_d = -2.225$ MeV", color = "navy")
plt.plot(S3, sig3, label = "$\epsilon_d = -2.325$ MeV", color = "forestgreen")
plt.plot(np.repeat(pos1,100), np.linspace(0, sig1.max(), 100), color = "firebrick")
plt.plot(np.repeat(pos2,100), np.linspace(0, sig2.max(), 100), color = "navy")
plt.plot(np.repeat(pos3,100), np.linspace(0, sig3.max(), 100), color = "forestgreen")
plt.xlabel("$S$ (MeV)", fontsize = 14)
plt.ylabel("d$^3 \sigma$/d$\Omega_1$d$\Omega_2$d$S$ (mb sr$^{-2}$ MeV$^{-1}$)", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.savefig("plots/phi0th355E13.pdf")

#theta = 43.0
S1, sig1 = np.loadtxt("txtfiles/phi0th43E13e125.txt", unpack = True, skiprows = 1)
pos1 = S1[0]
S1 = S1[1:]
sig1 = sig1[1:]
S2, sig2 = np.loadtxt("txtfiles/phi0th43E13e225.txt", unpack = True, skiprows = 1)
pos2 = S2[0]
S2 = S2[1:]
sig2 = sig2[1:]
S3, sig3 = np.loadtxt("txtfiles/phi0th43E13e325.txt", unpack = True, skiprows = 1)
pos3 = S3[0]
S3 = S3[1:]
sig3 = sig3[1:]

plt.figure()
plt.plot(S1, sig1, label = "$\epsilon_d = -2.125$ MeV", color = "firebrick")
plt.plot(S2, sig2, label = "$\epsilon_d = -2.225$ MeV", color = "navy")
plt.plot(S3, sig3, label = "$\epsilon_d = -2.325$ MeV", color = "forestgreen")
plt.plot(np.repeat(pos1,100), np.linspace(0, sig1.max(), 100), color = "firebrick")
plt.plot(np.repeat(pos2,100), np.linspace(0, sig2.max(), 100), color = "navy")
plt.plot(np.repeat(pos3,100), np.linspace(0, sig3.max(), 100), color = "forestgreen")
plt.xlabel("$S$ (MeV)", fontsize = 14)
plt.ylabel("d$^3 \sigma$/d$\Omega_1$d$\Omega_2$d$S$ (mb sr$^{-2}$ MeV$^{-1}$)", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.savefig("plots/phi0th43E13.pdf")
