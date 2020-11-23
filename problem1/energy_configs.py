import numpy as np
import matplotlib.pyplot as plt



def energy_configs(J, h, N, n):
    configs = np.zeros((n, N))

    for j in range(n):
        configs[j] = np.random.choice([-1, 1], N)
        
    def H(J, h, s):
        interact = np.sum(s[:-1]*s[1:]) + s[0]*s[-1]
        if N == 2:
            interact /= 2
        interact *= -J
        external = -h*np.sum(s)
        return interact + external
    
    energies = np.zeros(n)
    for i, config in enumerate(configs):
        energies[i] = H(J, h, config)
    
    return energies


def E_dist(N, alpha):
    energies = energy_configs(1, 0.5, N, int(alpha*2**N))
    E, count = np.unique(energies, return_counts=True)
    count = np.array(count)/np.sum(count)

    plt.scatter(E, count, label = str(alpha))

for alpha in [1, 10, 100]:
    E_dist(10, alpha)

plt.show()

