from orbital_elements import elements
from utils import visualization
from orbital_elements import tle
from propagators import analytical_propagators, cowell_propagators
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 5600, 10000)
earth_radius = 6378.0  # in km
mu = 3.986e5

tle1 = ["1 25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
        "2 25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"
       ]

state_vector_0 = tle.TLE_to_state_vectors(tle1, mu)
oes_0 = elements.get_osculating_elements(state_vector_0, mu)
oes_tle_0 = tle.TLE_to_orbital_elements(tle1, mu)

# print(state_vector_0)
print(f"a     :  {oes_tle_0[0]:.5f} {oes_0[0]:.5f}")
print(f"e     :  {oes_tle_0[1]:.5f} {oes_0[1]:.5f}")
print(f"i     :  {oes_tle_0[2]:.5f} {oes_0[2]:.5f}")
print(f"Omega :  {oes_tle_0[3]:.5f} {oes_0[3]:.5f}")
print(f"omega :  {oes_tle_0[4]:.5f} {oes_0[4]:.5f}")
print(f"theta :  {oes_tle_0[5]:.5f} {oes_0[5]:.5f}")