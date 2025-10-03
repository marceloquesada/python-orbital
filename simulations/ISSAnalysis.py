from orbital_elements import oeOpsArray
from utils import stateVectorsOperations
from utils import visualization
from utils import types
from utils import TLE
from propagators import analyticalPropagators, cowellPropagators
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 5600, 10000)
earth_radius = 6378.0  # in km
mu = 3.986e5

tle = ["25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
       "25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"
       ]

state_vector_0 = TLE.TLE_to_state_vectors(tle, mu)

propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector_0, mu)
t_analit, X_I_analit = propagator_analit.propagate(step_size=0.1)
oes_analit = propagator_analit.to_orbital_elements()

print("Running numerical propagator")
propagator_num = cowellPropagators.TwoBodyPropagator(state_vector_0, mu)
t_num, X_I_num = propagator_num.propagate(t, periods=1, integration_method='Radau')
oes_num = propagator_num.to_orbital_elements()

visualization.plot_3D_overlay(X_I_analit, X_I_num, orbit_marker='--')

visualization.plot_classic_orbital_elements_overlay([t_analit, oes_analit], [t_num, oes_num])

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
labels = ['x', 'y', 'z']
for i in range(3):
    axs[i].plot(t_analit, X_I_analit[i, :], label='Analytical')
    axs[i].plot(t_num, X_I_num[i, :], label='Numerical', linestyle='--')
    axs[i].set_ylabel(f'{labels[i]} (km)')
    axs[i].legend()
axs[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()