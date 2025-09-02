from propagators.cowellPropagators import TwoBodyPropagator
from utils.visualization import plot_3D_view, plot_classic_orbital_elements
import numpy as np

r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 200000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5

initial_state_vectors = np.concatenate((r, v))
propagator = TwoBodyPropagator(initial_state_vectors, mu)
t_revised, state_vectors = propagator.propagate(t, periods=1)
orbital_elementss = propagator.to_orbital_elements()

plot_3D_view(X=state_vectors)
plot_classic_orbital_elements(t_revised, orbital_elementss)
