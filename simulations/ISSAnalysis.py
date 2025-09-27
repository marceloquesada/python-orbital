from utils import visualization, stateVectors, orbitalElementsOperations
from utils import types
from propagators import analyticalPropagators, cowellPropagators
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 5600, 1000000)
earth_radius = 6378.0  # in km
mu = 3.986e5

orbital_elements_0 = types.OrbitalElements(
    major_axis=6793,
    eccentricity=0.0004379,
    inclination=51.6338,
    ascending_node=192.8556,
    argument_of_perigee=1.5470,
    true_anomaly=50
)
state_vector_0 = stateVectors.get_state_vectors(orbital_elements_0, mu)

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