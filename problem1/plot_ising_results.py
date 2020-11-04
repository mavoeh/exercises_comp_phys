import numpy as np
import matplotlib.pyplot as plt

h, N, Z, Z_analytical, delta, m = np.loadtxt("results_backup.txt", unpack = True)


data = np.zeros((19, 201, 4))

for n in range(19):
    mask = [N == n+2]
    data[n,:,:] = np.column_stack([Z[mask], Z_analytical[mask], delta[mask], m[mask]])

def hplot(i, N = 20):
    h = np.linspace(-1, 1, 201)
    plt.plot(h, data[N-2,:,i])

def Nplot(h):
	N = np.linspace(2, 20, 19)
	mh = np.zeros(19)
	for j in range(19):
		mh[j] = m[int((N[j]-2)*201+(h+1)*100)]
	plt.plot(N, mh)

for n in range(2,21):
    hplot(3, N=n)

plt.show()

for n in range(21):
	Nplot(-1+0.1*n)


plt.show()
