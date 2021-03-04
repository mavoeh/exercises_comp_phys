import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyScattering import *
from scurve import *
from fsi import *

def E0p(ed, ednew, E0, theta1, theta2, phi):
	c1, s1, c2, s2 = np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2)
	z = np.cos(phi)*s1*s2+c1*c2
	return (ed - ednew)*(4.-z**2)/2./(c1**2+c2**2-z*c1*c2) + E0


para=[700.0, 0.020167185806378923]
pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])

ed, _, _, _ = TwoBody(pot, np1=20, np2=10).esearch()

theta1 = 41.9
theta2 = 41.9
phi12 = 180.0
Elab = 22.7

scattL0 = ThreeBodyScatt(pot,e3n=(2./3.*Elab/ThreeBodyScatt.hbarc + ed),nx=16,np1=20,np2=10,nq1=15,nq2=5,lmax=0,bl=0)
ed *= scattL0.hbarc

S, kx, ky, t = scurve(theta1, theta2, phi12, Elab, e = ed, deg=True)
S = S[::1000]
kx = kx[::1000]
ky = ky[::1000]

sigma = np.zeros(len(S))

for i in range(len(S)):
  sigma[i] = scattL0.breakup_cross(Elab, kx[i], ky[i], theta1, theta2 ,phi12)

plt.plot(S, sigma, 'o-')
plt.show()

theta = 45.0
Elab = 30.0
ednew = -1.225

#Sfsi, sigma, peakpos = fsi(scattL0, theta, Elab, ed, deg=True)
#S2, sig2, peakpos2 = fsi(scattL0, theta, E0p(ed, ednew, Elab, theta, theta, 0.), ednew, deg = True)

'''
plt.plot(Sfsi, sigma, 'o-')
plt.plot(S2, sig2, 'o-')
plt.show()

print(peakpos, Sfsi[-1])
print(peakpos2, S2[-1])
'''
