import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from scurve import *

para=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

theta1 = 41.9
theta2 = 41.9
phi12 = 0.0
Elab = 22.7

scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab+ed)/ThreeBodyScatt.hbarc,nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)

S, kx, ky, t = scurve(theta1, theta2, phi12, Elab, deg=True)
S = S[::1000]
kx = kx[::1000]
ky = ky[::1000]
#print(S[-1])

sigma = np.zeros(len(S), dtype = np.double)

for i in range(0, len(S)):
  sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)

plt.plot(S, sigma, 'o-')
plt.show()

plt.plot(S[2:-2], sigma[2:-2])
plt.show()