from orbital_elements import oeOpsArray
from utils import stateVectorsOperations
from utils import visualization
from utils import types
from utils import TLE
from propagators import analyticalPropagators, cowellPropagators
import numpy as np
import matplotlib.pyplot as plt
import requests

t = np.linspace(0, 5600, 10000)
earth_radius = 6378.0  # in km
mu = 3.986e5

def process_tle():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Encountered error {response.status_code} while trying to access Celestrak")
    else:
        print("TLEs OK!")

    tle_body = response.text
    tle_lines = tle_body.split("\n")
    tle_lines = [line.split("\r")[0] for line in tle_lines]

    print(f"Got {len(tle_lines)//3} TLEs")

    Xs = []

    for i in range(len(tle_lines)//3):
        print(f"Processing {tle_lines[3*i].split("        ")[0]}")
        state_vector_0 = TLE.TLE_to_state_vectors(tle_lines[3*i+1:3*i+3], mu)

        propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector_0, mu)
        t_analit, X_I_analit = propagator_analit.propagate(step_size=0.1)

        Xs.append(X_I_analit)

    visualization.plot_3D_overlay(*Xs)

process_tle()



# tle = ["1 25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
#        "2 25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"
#        ]

# state_vector_0 = TLE.TLE_to_state_vectors(tle, mu)

# propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector_0, mu)
# t_analit, X_I_analit = propagator_analit.propagate(step_size=0.1)
# oes_analit = propagator_analit.to_orbital_elements()

# visualization.plot_3D_view(X_I_analit)