from utils import visualization
from propagators import analytical_propagators
import numpy as np


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 200000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5

state_vector_0 = np.concatenate((r, v))

print(state_vector_0.shape)
propagator = analytical_propagators.Two_body_analytical_propagator(state_vector_0, mu)
t, X_I = propagator.propagate(step_size=1)

visualization.plot_3D_view(t, X_I)
