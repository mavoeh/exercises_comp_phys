import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from fit import *
from scurve import *
from fsi import *

m = 938.92 #MeV

'''
para2=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para2[1],A=-1.0/6.474860194946856,cutoff=para2[0])

ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

theta1 = 41.9
theta2 = 41.9
phi12 = 0.
Elab = 22.7

scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
ed *= scattL0.hbarc

S, _, kx, ky, t = scurve(theta1, theta2, phi12, Elab, e = ed, deg=True)
Nc = 100
Ns = 10**5
S = S[::int(np.round(Ns/Nc))]
kx = kx[::int(np.round(Ns/Nc))]
ky = ky[::int(np.round(Ns/Nc))]

sigma = np.zeros(len(S))
for i in range(len(S)):
	sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)

plt.plot(S, sigma, 'o-')
plt.show()

'''

#For different parameter sets in parset, choose constant Elab and theta
Lam = [700.0]
e0list = [-2.125, -2.225, -2.325]
Elab = 13.
theta = 20.5

for e0 in e0list:
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])
	ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	plt.plot(S, sigma)
	#for j in range(len(pos)):
	plt.plot(np.repeat(pos[0],100), np.linspace(0, sigma.max(), 100))
	
plt.show() 

