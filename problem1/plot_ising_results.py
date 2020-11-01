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

for n in range(2,21):
    hplot(3, N=n)
    
plt.show()
