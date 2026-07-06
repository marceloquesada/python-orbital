# import sys
# sys.path.append('..//propagators')
# sys.path.append('..//frames')
# sys.path.append('../../orbital_elements')
# sys.path.append('..//utils')
# sys.path.append('..//visualization')
# sys.path.append('..//images')

# print(sys.path)

from python_orbital.orbital_elements import (
    oeOps,
    oeOpsArray,
    stateVectorOps
)
from ..utils import (
    types,
    constants
)
from ..visualization import (
    plots3D
)
from ..propagators import (
    analyticalPropagators
)
from ..TLE import *

import numpy as np
import matplotlib.pyplot as plt
import requests

t = np.linspace(0, 5600, 10000)
earth_radius = 6378.0  # in km
mu = 3.986e5


names, state_vectors = get_process_TLE(mu, "goes")

Xss = []

for state_vector in state_vectors:
    propagator_analit = analyticalPropagators.TwoBodyAnalyticalPropagator(state_vector, mu)
    t_analit, X_I_analit = propagator_analit.propagate(step_size=0.1)

    Xss.append(X_I_analit)


orbital_datas = []

for i in range(len(Xss)):
    orbital_data = types.PlotOrbitData(
        satellite_name=names[i],
        Xs=Xss[i],
    )
    orbital_datas.append(orbital_data)

plots3D.plot_3D_overlay(*orbital_datas)