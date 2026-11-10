from propagators import cowell_propagators, analytical_propagators
from perturbations import atmospheric_drag
from orbital_utils import constants
from orbital_elements import tle, state_vectors, elements
from utils import csv, visualization
import datetime

import numpy as np


gamma = 5  # Area-to-mass ratio [m^2/kg]
drag_coef = 2.2
f_x = 150  # Daily F10.7 solar flux [sfu]
f_0 = 150  # 81-day mean F10.7 solar flux [sfu]
k_p = 4    # Geomagnetic index (converted to ap = 27 for MSIS)
# epoch = datetime.datetime(1, 3, 21, 15, 0, 0, tzinfo=datetime.timezone.utc).timestamp()  # Day 80, 15h UTC
epoch = 0

t_initial = 0
t_max_days = 65
t_step = 50
decimation = 1

t = np.arange(t_initial, t_max_days*(24*3600), t_step)

oes_0 = np.array([7100, 0.001, 0.0002, 200, 200, 0])
state_vector_0 = state_vectors.get_state_vectors(oes_0, constants.mu_earth)
# state_vector_0 = np.array([-6077.3617217084,-739.7225171285,-2712.6578881687,-2.2292253536,-4.1335673096,6.1212036073])

# NUMERICAL PROPAGATOR
# drag_func = atmospheric_drag.Atmospheric_drag_perturbation_td88(drag_coef, gamma, f_x, f_0, k_p, epoch)
drag_func = atmospheric_drag.Atmospheric_drag_perturbation_msis(drag_coef, gamma, f_x, f_0, atmospheric_drag.kp_to_ap(k_p), epoch)
# drag_func = atmospheric_drag.Atmospheric_drag_perturbation_exp(drag_coef, gamma)
num_prop = cowell_propagators.Perturbed_propagator(state_vector_0, constants.mu_earth, [drag_func])

t_num, X_num = num_prop.propagate(t, integration_method="RK45")

oes_num = num_prop.to_orbital_elements()

visualization.plot_classic_orbital_elements_overlay([t_num/(24*3600), oes_num])

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

csv.save_orbit(t_f, X_ff, append=False)