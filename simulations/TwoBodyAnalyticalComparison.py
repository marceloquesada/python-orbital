from utils import visualization
from propagators import analyticalPropagators, cowellPropagators
import numpy as np


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 200000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5

state_vector_0 = np.concatenate((r, v))

propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector_0, mu)
X_I_analit = propagator_analit.propagate(step_size=0.1)
oe_analit = propagator_analit.orbital_elements

propagator_num = cowellPropagators.TwoBodyPropagator(state_vector_0, mu)
t_new, X_I_num = propagator_num.propagate(t, periods=1)
oes_num = propagator_num.to_orbital_elements()

oes_analit = len(oes_num)*[oe_analit]

visualization.plot_3D_overlay(X_I_analit, X_I_num, orbit_marker='--')

visualization.plot_classic_orbital_elements_overlay([t_new, oes_analit], [t_new, oes_num])
