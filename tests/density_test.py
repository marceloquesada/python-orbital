from perturbations import atmospheric_drag
import numpy as np
import matplotlib.pyplot as plt


hs = np.arange(0, 1000, 1)
densities = []

for h in hs:
    density = atmospheric_drag.simple_exponential_density_model(h)
    densities.append(density)

plt.plot(hs, densities)
plt.yscale('log')
plt.show()
    

