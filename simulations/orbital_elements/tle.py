import numpy as np
import re

from simulations.orbital_elements import state_vectors
from simulations.orbital_elements import elements


def TLE_to_orbital_elements(TLE: list[str], mu) -> np.typing.NDArray:
    elements_tle = re.split(" +", TLE[1])

    i = float(elements_tle[2])
    Omega = float(elements_tle[3])
    e = float('0.' + elements_tle[4])
    omega = float(elements_tle[5])
    M = float(elements_tle[6])

    # Getting semi-major axis from mean motion
    mean_motion = float(elements_tle[7])
    n = mean_motion * ((2*np.pi) / 86400)
    a = np.cbrt(mu/(n**2))

    # Getting true anomaly from mean anomaly
    theta = elements.get_true_anomaly_from_mean(e, M)

    oe = np.array([a, i, Omega, omega, e, theta])

    return  oe


def TLE_to_state_vectors(TLE: list[str], mu) -> np.typing.NDArray:
    orbital_elements = TLE_to_orbital_elements(TLE, mu)

    state_vectors = state_vectors.get_state_vectors(orbital_elements, mu)

    return state_vectors