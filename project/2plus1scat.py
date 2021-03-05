import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from scurve import *
from fsi import *

m = 938.92 #MeV

para=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

theta1 = 41.9
theta2 = 41.9
phi12 = 180.
Elab = 22.7

scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
ed *= scattL0.hbarc


S, _, kx, ky, t = scurve(theta1, theta2, phi12, Elab, e = ed, deg=True)
S = S[::1000]
kx = kx[::1000]
ky = ky[::1000]

sigma = np.zeros(len(S))
kin = np.zeros(len(S))

#k1k2 = np.sin(theta1)*np.sin(theta2)*np.cos(phi12)+ np.cos(theta2)*np.cos(theta1)
#k0 = np.sqrt(2*Elab*m)
#q0 = 2./3.*k0
for i in range(len(S)):
	sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)


plt.plot(S[:-4], sigma[:-4], 'o-')
plt.show()


'''
For different parameter sets in parset, choose constant Elab and theta
for i in range(3):
	para = parset[i]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])
	ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	plt.plot(S, sigma)
	for j in range(len(pos)):
		plt.plot(np.repeat(pos[j],100), np.linspace(sigma.max(), 100))
	plt.show() 

'''