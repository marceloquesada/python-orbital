from propagators import cowell_propagators, analytical_propagators
from perturbations import earth_oblateness
from orbital_utils import constants
from orbital_elements import elements, state_vectors, tle
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


t = np.arange(0, 4e5, 5)

# Eccentricity must be well above the J2 short-period oscillation of e (~1e-3),
# otherwise the osculating argument of perigee is dominated by that oscillation

state_vector_0 = tle.TLE_to_state_vectors(
    ["1 04382U 70034A   26311.23270489  .00001391  00000+0  19820-3 0  9995",
    "2 04382  68.4252 129.8277 1030755 211.4200 142.1156 13.14727277664482"], constants.mu_earth)

# ANALYTICAL PROPAGATOR
analit_prop = analytical_propagators.Two_body_analytical_propagator(state_vector_0, constants.mu_earth)
t_analit, X_analit = analit_prop.propagate()
oes_analit = analit_prop.to_orbital_elements()

# NUMERICAL PROPAGATOR
j2_func = earth_oblateness.J2_perturbation()
num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [j2_func])
t_num, X_num = num_prop.propagate(t, integration_method='Radau')

print(X_analit.shape)
print(X_num.shape)

oes_num = num_prop.to_orbital_elements()


oes_j2_comparison = np.zeros_like(oes_num)


Omega_0 = oes_num[3, 0]
Omegadot = np.rad2deg(get_analitical_raan_drift(state_vector_0))
for i in range(oes_num.shape[1]):
    oes_j2_comparison[3, i] = Omega_0 + t_num[i]*Omegadot

# visualization.plot_3D_overlay(X_analit, X_num)
# visualization.plot_classic_orbital_elements_overlay([t_analit, oes_analit], [t_num, oes_num], [t_num, oes_j2_comparison])

plt.plot(t_num, oes_num[3, :], lw=0.5, label='Simulação numérica')
plt.plot(t_num, oes_j2_comparison[3, :], lw=0.5, label='Desvio analítico esperado')
plt.title('Efeito de perturbação J2')
plt.xlabel('Tempo (s)')
plt.ylabel('Ascenção reta do nodo ascendente (º)')
plt.grid(True)
plt.legend()
plt.show()

oes_j2_comparison = np.zeros_like(oes_num)

omega_0 = oes_num[4, 0]
omegadot = np.rad2deg(get_analitical_argument_of_perigee_drift(state_vector_0))
for i in range(oes_num.shape[1]):
    oes_j2_comparison[4, i] = omega_0 + t_num[i]*omegadot

plt.plot(t_num, oes_num[4, :], lw=0.5, label='Simulação numérica')
plt.plot(t_num, oes_j2_comparison[4, :], lw=0.5, label='Desvio analítico esperado')
plt.title('Efeito de perturbação J2')
plt.xlabel('Tempo (s)')
plt.ylabel('Argumento do perigeu (º)')
plt.grid(True)
plt.legend()
plt.show()