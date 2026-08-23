from orbital_elements import oeOpsArray
from orbital_elements import stateVectorOps
from utils import visualizations
from utils import types
from visualization import plots3D
from TLE import TLEOps
from propagators import analyticalPropagators, Cowell_propagators
import numpy as np
import matplotlib.pyplot as plt
import requests

t = np.linspace(0, 5600, 10000)
earth_radius = 6378.0  # in km
mu = 3.986e5

def process_tle(satellite_group: str = "goes") -> list[np.typing.NDArray]:
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={satellite_group}&FORMAT=tle"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Encountered error {response.status_code} while trying to access Celestrak")
    else:
        print("TLEs OK!")

    tle_body = response.text
    tle_lines = tle_body.split("\n")
    tle_lines = [line.split("\r")[0] for line in tle_lines]

    state_vectors = []

    for i in range(len(tle_lines)//3):
        print(f"Processing {tle_lines[3*i].split("        ")[0]}")
        state_vector = TLEOps.TLE_to_state_vectors(tle_lines[3*i+1:3*i+3], mu)

        state_vectors.append(state_vector)

    return state_vectors

state_vectorss = process_tle()

Xs = []

for state_vector in state_vectorss:
    propagator_num = Cowell_propagators.TwoBodyPropagator(state_vector, mu)
    t_analit, X_I_analit = propagator_num.propagate(np.linspace(0, 2000000, 10000),periods=1)

    Xs.append(X_I_analit)

plots3D.plot_3D_overlay(*Xs)