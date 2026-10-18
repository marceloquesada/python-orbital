from propagators import cowell_propagators, analytical_propagators
from perturbations import atmospheric_drag
from orbital_utils import constants
from orbital_elements import tle
from utils import csv, visualization

import numpy as np


contact_area = 0.1**2  # Lateral area for cubesat
mass = 5
drag_coef = 2.2

t = np.arange(0, 252288000, 50)

tle1 = ["1 25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
        "2 25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"
       ]

state_vector_0 = tle.TLE_to_state_vectors(tle1, constants.mu_earth)

# ANALYTICAL PROPAGATOR
analit_prop = analytical_propagators.Two_body_analytical_propagator(state_vector_0, constants.mu_earth)
t_analit, X_analit = analit_prop.propagate()
oes_analit = analit_prop.to_orbital_elements()

# NUMERICAL PROPAGATOR
drag_func = atmospheric_drag.Atmospheric_drag_perturbation(contact_area, drag_coef, mass, optimization_steps=100)

num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [drag_func])
# num_prop = cowell_propagators.Drag_propagator(state_vector_0, constants.mu_earth, mass, contact_area, drag_coef)

t_num, X_num = num_prop.propagate(t, integration_method="Radau")

csv.save_orbit(t_num, X_num)


# oes_num = num_prop.to_orbital_elements()

# visualization.plot_3D_overlay(X_analit, X_num)
# visualization.plot_classic_orbital_elements_overlay([t_analit, oes_analit], [t_num, oes_num])
