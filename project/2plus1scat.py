import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from fit import *
from scurve import *
from fsi import *

m = 938.92 #MeV


para2=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para2[1],A=-1.0/6.474860194946856,cutoff=para2[0])

ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

theta1 = 39.
theta2 = 39.
phi12 = 180.
Elab = 13.

scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
ed *= scattL0.hbarc

theta = np.array([theta1])

melast = scattL0.get_m_elast_drive(theta) + scattL0.get_m_elast_pw(theta)

sigma = (2*np.pi)**4*2./3.*(m/scattL0.hbarc)**2*np.absolute(melast)**2/scattL0.hbarc*10.
print(sigma)

'''
S, _, kx, ky, t = scurve(theta1, theta2, phi12, Elab, e = ed, deg=True)
Nc = 1000
Ns = 10**5+1
S = S[::int(np.round(Ns/Nc))]
kx = kx[::int(np.round(Ns/Nc))]
ky = ky[::int(np.round(Ns/Nc))]

f = open("phi180th39E13.txt", "w")
f.write("phi12 = 180°, theta1 = theta2 = 39°, Elab = 13 MeV. Compare to Glöckle S.189\n")
sigma = np.zeros(len(S))
for i in range(len(S)):
	sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)
	# write sigma and S to a text file
	f.write("{0:e}\t{1:e}\n".format(S[i], sigma[i]))
f.close()

plt.plot(S, sigma, 'o-')
plt.show()


#For different parameter sets in parset (i.e. different binding energies), 
#vary Elab and theta
Lam = [700.0]
e0list = [-2.125, -2.225, -2.325]
Elab = 13.
theta = 43.0

names = ["txtfiles/phi0th43E13e125.txt", "txtfiles/phi0th43E13e225.txt", "txtfiles/phi0th43E13e325.txt"]

for i, e0 in enumerate(e0list):
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])
	ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	f = open(names[i], "w")
	f.write("phi12 = 0°, theta1 = theta2 = 43.0°, Elab = 13 MeV, ed = {0:.3f}. Compare to Trotter\n".format(e0))
	f.write("{0:e}\t{1:e}\n".format(pos[0], 0.0))
	for j in range(len(S)):
		f.write("{0:e}\t{1:e}\n".format(S[j], sigma[j]))
	f.close()

	plt.plot(S, sigma)
	#for j in range(len(pos)):
	plt.plot(np.repeat(pos[0],100), np.linspace(0, sigma.max(), 100))
	
plt.show() 
'''
