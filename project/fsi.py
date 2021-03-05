import numpy as np
from scurve import scurve
import matplotlib.pyplot as plt

def fsi(threeBodySolver, theta, Elab, e, deg = True,
        m = 938.92,
        N_scurve = 10**5+1,
        Nc = 100):
    
    # get S curve for given parameters
    S, Sk, k1, k2, t = scurve(theta, theta, 0, Elab, e = e, m = m, N = N_scurve, deg = deg)
    
    # list in which the S values of the expected FSI peaks will be stored
    fsi_peakpos = []
    
    # Find the values of S for which FSI peaks are expected
    for ti in [np.pi/2, 3*np.pi/2]:
        i = np.argmin( np.abs(t%(2*np.pi) - ti) )
        if np.abs(t[i]%(2*np.pi) - ti) <= np.pi/N_scurve:
            fsi_peakpos.append(S[i])

    
    # now plot the cross section (but on a coarser grid)
    Ns = len(S)
    S = S[::int(np.round(Ns/Nc))]
    Sk = Sk[::int(np.round(Ns/Nc))]
    k1 = k1[::int(np.round(Ns/Nc))]
    k2 = k2[::int(np.round(Ns/Nc))]
    sig = np.zeros(len(S)) # array for the cross section
    for i in range(len(S)):
        sig[i] = threeBodySolver.breakup_cross(Elab, k1[i], k2[i], theta, theta, 0)
    
    
    return S, Sk, sig, fsi_peakpos    