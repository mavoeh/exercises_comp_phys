import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from fit import *
from scurve import *
from fsi import *
import pickle


m = 938.92 #MeV

#chosen parameters for plots
np1 = 20
np2 = 10
nq1 = 20
nq2 = 10
pc = 20
qc = 20


#Stability Test here: No good resulst yet!

Lam = [700.0]
e0list = [-2.125, -2.225, -2.325]

para = fitc0(Lam,e0list[1])[0]
pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

Elab = 13.
theta = 43.0
phi12 = 0

sig = np.zeros((8,8))
for i, np1 in enumerate([20, 22, 24, 26, 28, 30, 32, 34]):
	for j, nq1 in enumerate([20, 22, 24, 26, 28, 30, 32, 34]):
		ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
		scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)

		S, _, kx, ky, t = scurve(theta, theta, phi12, Elab, e = ed, deg=True)
		it = np.abs(t-np.pi/2.).argmin()
		sig[i,j] = scattL0.breakup_cross(Elab, kx[it], ky[it], theta, theta ,phi12)

		print("np1={0:d},np2={1:d},nq1={2:d},nq2={3:d},pc={4:d},qc={5:d}\t{6:e}\n".format(np1, np2, nq1, nq2, pc, qc, sig[i,j]))
		pickle.dump(sig, open("txtfiles/cross_sec_convergence.p", "wb"))


'''
#random configuration
para2=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para2[1],A=-1.0/6.474860194946856,cutoff=para2[0])

theta1 = 10.0
theta2 = 45.0
phi12 = 45.
Elab = 25.

ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)
ed *= scattL0.hbarc

S, _, kx, ky, t = scurve(theta1, theta2, phi12, Elab, e = ed, deg=True)
Nc = 1000
Ns = 10**5+1
S = S[::int(np.round(Ns/Nc))]
kx = kx[::int(np.round(Ns/Nc))]
ky = ky[::int(np.round(Ns/Nc))]

f = open("txtfiles/phi45th1045E25.txt", "w")
f.write("phi12 = 45°, theta1 = 10°, theta2 = 45.0°, Elab = 25 MeV \n")
sigma = np.zeros(len(S))
for i in range(len(S)):
	sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)
	# write sigma and S to a text file
	f.write("{0:e}\t{1:e}\n".format(S[i], sigma[i]))
f.close()



#For different parameter sets in parset (i.e. different binding energies), 
#vary theta

Lam = [700.0]
e0list = [-2.125, -2.225, -2.325]
Elab = 13.

theta = 20.5
names = ["txtfiles/phi0th205E13e125.txt", "txtfiles/phi0th205E13e225.txt", "txtfiles/phi0th205E13e325.txt"]

for i, e0 in enumerate(e0list):
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

	ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	f = open(names[i], "w")
	f.write("phi12 = 0°, theta1 = theta2 = 20.5°, Elab = 13 MeV, ed = {0:.3f}. Compare to Trotter\n".format(e0))
	f.write("{0:e}\t{1:e}\n".format(pos[0], 0.0))
	for j in range(len(S)):
		f.write("{0:e}\t{1:e}\n".format(S[j], sigma[j]))
	f.close()


theta = 28.0
names = ["txtfiles/phi0th28E13e125.txt", "txtfiles/phi0th28E13e225.txt", "txtfiles/phi0th28E13e325.txt"]

for i, e0 in enumerate(e0list):
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

	ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	f = open(names[i], "w")
	f.write("phi12 = 0°, theta1 = theta2 = 28.0°, Elab = 13 MeV, ed = {0:.3f}. Compare to Trotter\n".format(e0))
	f.write("{0:e}\t{1:e}\n".format(pos[0], 0.0))
	for j in range(len(S)):
		f.write("{0:e}\t{1:e}\n".format(S[j], sigma[j]))
	f.close()


theta = 35.5
names = ["txtfiles/phi0th355E13e125.txt", "txtfiles/phi0th355E13e225.txt", "txtfiles/phi0th355E13e325.txt"]

for i, e0 in enumerate(e0list):
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

	ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	f = open(names[i], "w")
	f.write("phi12 = 0°, theta1 = theta2 = 35.5°, Elab = 13 MeV, ed = {0:.3f}. Compare to Trotter\n".format(e0))
	f.write("{0:e}\t{1:e}\n".format(pos[0], 0.0))
	for j in range(len(S)):
		f.write("{0:e}\t{1:e}\n".format(S[j], sigma[j]))
	f.close()
 

theta = 43.0
names = ["txtfiles/phi0th43E13e125.txt", "txtfiles/phi0th43E13e225.txt", "txtfiles/phi0th43E13e325.txt"]

for i, e0 in enumerate(e0list):
	para = fitc0(Lam,e0)[0]
	pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

	ed, _, _, _ = TwoBody(pot, np1=np1, np2=np2, pc=pc).esearch()
	scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=np1,np2=np2,nq1=nq1,nq2=nq2,pc=pc,qc=qc,lmax=0,bl=0)
	ed *= scattL0.hbarc

	S, _, sigma, pos = fsi(scattL0, theta, Elab, ed)

	f = open(names[i], "w")
	f.write("phi12 = 0°, theta1 = theta2 = 43.0°, Elab = 13 MeV, ed = {0:.3f}. Compare to Trotter\n".format(e0))
	f.write("{0:e}\t{1:e}\n".format(pos[0], 0.0))
	for j in range(len(S)):
		f.write("{0:e}\t{1:e}\n".format(S[j], sigma[j]))
	f.close()
'''