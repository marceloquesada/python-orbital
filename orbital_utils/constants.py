import numpy as np


MIN_ALLOWED_VALUE = 1e-10

class Bases:
    i: np.typing.NDArray = np.array([1, 0, 0])
    j: np.typing.NDArray = np.array([0, 1, 0])
    k: np.typing.NDArray = np.array([0, 0, 1])

mass_earth = 5.97219e24  # Kg

## EARTH
radius_earth = 6378.137 # Km
mu_earth = 3.986004418e5
w_earth = 7.2921150e-5 # rad/s
j2_earth = 1082.64e-6