from propagators import cowell_propagators, analytical_propagators
from perturbations import atmospheric_drag
from orbital_utils import constants
from orbital_elements import tle, state_vectors, elements
from utils import csv, visualization

import numpy as np


mass = 5
contact_area = (0.1**2)*mass  # Lateral area for cubesat
drag_coef = 2.2

parts = 1000
t_initial = 0
t_max = 1e9
t_step = 50
decimation = 100

t = np.arange(t_initial, t_max, t_step)

oes_0 = np.array([500+constants.radius_earth, 0.00001, 30, 200, 200, 0])
state_vector_0 = state_vectors.get_state_vectors(oes_0, constants.mu_earth)
# state_vector_0 = np.array([-6077.3617217084,-739.7225171285,-2712.6578881687,-2.2292253536,-4.1335673096,6.1212036073])

for i in range(parts):

        t_part = t[round((len(t)/parts)*i):round((len(t)/parts)*(i+1))]

        # NUMERICAL PROPAGATOR
        drag_func = atmospheric_drag.Atmospheric_drag_perturbation(contact_area, drag_coef, mass)
        num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [drag_func])

        t_num, X_num = num_prop.propagate(t_part, integration_method="Radau")

        state_vector_0 = X_num[:,-1]

        X_f = np.empty_like(X_num)
        t_f = []
        dec_counter = 0
        j = 0
        for i in range(len(t_num)):
                if dec_counter >= decimation:
                        t_f.append(t_num[i])
                        X_f[:, j] = X_num[:, i]
                        dec_counter = 0
                        j += 1
                else:
                        dec_counter+=1

        X_ff = X_f[:, :j]
        if False:
                append = False
        else:
                append = True


        print(X_ff.shape)

        csv.save_orbit(t_f, X_ff, append=append)