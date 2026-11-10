from propagators import cowell_propagators, analytical_propagators
from perturbations import earth_oblateness
from orbital_utils import constants
from orbital_elements import elements, state_vectors
from utils import visualization

import matplotlib.pyplot as plt
import numpy as np


def get_analitical_raan_drift(state_vector):
    oe = elements.get_osculating_elements(state_vector, constants.mu_earth)
    p0 = oe[0]*(1 - oe[1]**2)
    n = np.sqrt(constants.mu_earth/(oe[0]**3))
    # n = (1 + 1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*np.sqrt(1 - oe[1]**2)*(1 - 1.5*(np.sin(np.deg2rad(oe[2]))**2)))*np.sqrt(constants.mu_earth/(oe[0]**3))
    Omegadot = (-1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*np.cos(np.deg2rad(oe[2])))*n
    
    return Omegadot

def get_analitical_argument_of_perigee_drift(state_vector):
    oe = elements.get_osculating_elements(state_vector, constants.mu_earth)
    p0 = oe[0]*(1 - oe[1]**2)
    n = np.sqrt(constants.mu_earth/(oe[0]**3))
    # n = (1 + 1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*np.sqrt(1 - oe[1]**2)*(1 - 1.5*(np.sin(np.deg2rad(oe[2]))**2)))*np.sqrt(constants.mu_earth/(oe[0]**3))
    omegadot = (1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*(2 - 2.5*(np.sin(np.deg2rad(oe[2]))**2)))*n
    
    return omegadot


t = np.arange(0, 1e5, 5)

# Eccentricity must be well above the J2 short-period oscillation of e (~1e-3),
# otherwise the osculating argument of perigee is dominated by that oscillation
oe_0 = np.array([7000, 0.1, 51.6327, 135.3521, 189.2069, 170.892])  # a, e, i, Omega, omega, theta

state_vector_0 = state_vectors.get_state_vectors(oe_0, constants.mu_earth)

# ANALYTICAL PROPAGATOR
analit_prop = analytical_propagators.Two_body_analytical_propagator(state_vector_0, constants.mu_earth)
t_analit, X_analit = analit_prop.propagate()
oes_analit = analit_prop.to_orbital_elements()

# NUMERICAL PROPAGATOR
j2_func = earth_oblateness.J2_perturbation()
num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [j2_func])
t_num, X_num = num_prop.propagate(t)

print(X_analit.shape)
print(X_num.shape)

oes_num = num_prop.to_orbital_elements()

oes_j2_comparison = np.zeros_like(oes_num)

Omega_0 = oes_num[3, 0]
omega_0 = oes_num[4, 0]
Omegadot = np.rad2deg(get_analitical_raan_drift(state_vector_0))
omegadot = np.rad2deg(get_analitical_argument_of_perigee_drift(state_vector_0))
for i in range(oes_num.shape[1]):
    oes_j2_comparison[3, i] = Omega_0 + t_num[i]*Omegadot
    oes_j2_comparison[4, i] = omega_0 + t_num[i]*omegadot

# Running moving average over one orbital period (removes J2 short-period terms)
period = 2*np.pi*np.sqrt(oe_0[0]**3/constants.mu_earth)
dt = t_num[1] - t_num[0]
N = int(round(period/dt))
kernel = np.ones(N)/N

# Unwrap angles so a 360 -> 0 crossing doesn't corrupt the average
Omega_unwrapped = np.unwrap(oes_num[3, :], period=360)
omega_unwrapped = np.unwrap(oes_num[4, :], period=360)

mean_Omega = np.convolve(Omega_unwrapped, kernel, mode='valid')
mean_omega = np.convolve(omega_unwrapped, kernel, mode='valid')
# Time at the center of each averaging window (same length as the means)
t_mean = np.convolve(t_num, kernel, mode='valid')

# Analytical secular drift anchored at the first mean value, so the comparison
# isn't offset by the short-period oscillation present in the osculating Omega_0/omega_0
Omega_analit = mean_Omega[0] + (t_num - t_mean[0])*Omegadot
omega_analit = mean_omega[0] + (t_num - t_mean[0])*omegadot

print(f'Max |mean - analytical| Omega: {np.max(np.abs(mean_Omega - np.interp(t_mean, t_num, Omega_analit))):.4f} deg')
print(f'Max |mean - analytical| omega: {np.max(np.abs(mean_omega - np.interp(t_mean, t_num, omega_analit))):.4f} deg')

plt.plot(t_num, Omega_unwrapped, lw=0.5, label='Simulação numérica')
plt.plot(t_mean, mean_Omega, lw=1, label='Média móvel (1 período)')
plt.title('Efeito de perturbação J2')
plt.xlabel('Tempo (s)')
plt.ylabel('Ascenção reta do nodo ascendente (º)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t_num, omega_unwrapped, lw=0.5, label='Simulação numérica')
plt.plot(t_mean, mean_omega, lw=1, label='Média móvel (1 período)')
plt.title('Efeito de perturbação J2')
plt.xlabel('Tempo (s)')
plt.ylabel('Argumento do perigeu (º)')
plt.grid(True)
plt.legend()
plt.show()