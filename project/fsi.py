import numpy as np
from scurve import scurve
import matplotlib.pyplot as plt

def fsi(threeBodySolver, theta, Elab,
        e = -2.225,
        m = 938.92,
        N_scurve = 10**5+1,
        Nc = 10
        deg = False):
    
    # get S curve for given parameters
    S, k1, k2, t = scurve(theta, theta, 0, Elab, m = m, e = e, N = N_scurve, deg = deg)
    
    # list in which the S values of the expected FSI peaks will be stored
    fsi_peakpos = []
    
    # Find the values of S for which FSI peaks are expected
    for ti in [3*np.pi/2, np.pi/2]:
        i = np.argmin( np.abs(t%(2*np.pi) - ti) )
        if np.abs(t[i]%(2*np.pi) - ti) <= np.pi/N_scurve:
            fsi_peakpos.append(S[i])
    
    # now plot the cross section (but on a coarser grid)
    Ns = len(S)
    S = S[::np.round(Ns/Nc)]
    k1 = k1[::np.round(Ns/Nc)]
    k2 = k2[::np.round(Ns/Nc)]
    sig = np.zeros(Nc) # array for the cross section
    for i in range(Nc):
        sig[i] = threeBodySolver.breakup_cross(Elab, k1[i], k2[i], theta, theta, 0)
    
    

print(fsi(None, 30, 100, deg = True)[-1])
plt.gca().axis('equal')
plt.show()
    
