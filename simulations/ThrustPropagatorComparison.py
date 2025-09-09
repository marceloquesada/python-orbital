from utils import visualization
from propagators import analyticalPropagators, cowellPropagators
from perturbations import thrust_perturbations
import numpy as np
import matplotlib.pyplot as plt


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 50000, 1000000)
earth_radius = 6378.0  # in km
mu = 3.986e5

thrust = 100  # In Km/s
                #  r, v, h
thrust_direction = np.array([0, 1, 0])
Isp = 220
m_sat = 20

state_vector_0 = np.concatenate((r, v))


# Propagador analítico de 2 corpos
propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector_0, mu)
t_analit, X_I_analit = propagator_analit.propagate(step_size=0.1)
oes_analit = propagator_analit.to_orbital_elements()

# Propagador com empuxo
angle_intervs = [(179, 181)]
thrust_perturb = thrust_perturbations.FixedMassThetaIntervalThrust(thrust, thrust_direction, m_sat, mu, angle_intervs)
perturbs = [thrust_perturb]

propagator_num = cowellPropagators.PerturbedPropagator(state_vector_0, mu, perturbs)
t_num, X_I_num = propagator_num.propagate(t, periods=1)
oes_num = propagator_num.to_orbital_elements()
print(oes_analit[0].true_anomaly, oes_analit[-1].true_anomaly)

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
