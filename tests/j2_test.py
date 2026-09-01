from propagators import cowell_propagators, analytical_propagators
from perturbations import earth_oblateness
from orbital_utils import constants
from orbital_elements import tle, elements
from utils import visualization

import numpy as np


def get_analitical_raan_drift(state_vector):
    oe = elements.get_osculating_elements(state_vector, constants.mu_earth)
    p0 = oe[0]*(1 - oe[1]**2)
    n = np.sqrt(constants.mu_earth/(oe[0]**3))
    # n = (1 + 1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*np.sqrt(1 - oe[1]**2)*(1 - 1.5*(np.sin(np.deg2rad(oe[2]))**2)))*np.sqrt(constants.mu_earth/(oe[0]**3))
    Omegadot = (-1.5*constants.j2_earth*((constants.radius_earth/p0)**2)*np.cos(np.deg2rad(oe[2])))*n
    
    return Omegadot


t = np.arange(0, 1e5, 10)

tle1 = ["1 25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
        "2 25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"
       ]

state_vector_0 = tle.TLE_to_state_vectors(tle1, constants.mu_earth)

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
Omegadot = np.rad2deg(get_analitical_raan_drift(state_vector_0))
for i in range(oes_num.shape[1]):
    oes_j2_comparison[3, i] = Omega_0 + t_num[i]*Omegadot



visualization.plot_3D_overlay(X_analit, X_num)
visualization.plot_classic_orbital_elements_overlay([t_analit, oes_analit], [t_num, oes_num], [t_num, oes_j2_comparison])