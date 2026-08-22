import numpy as np

MIN_ALLOWED_VALUE = 1e-10

class Bases:
    i: np.typing.NDArray = np.array([1, 0, 0])[:, np.newaxis]
    j: np.typing.NDArray = np.array([0, 1, 0])[:, np.newaxis]
    k: np.typing.NDArray = np.array([0, 0, 1])[:, np.newaxis]

mass_earth = 5.97219e24  # Kg
radius_earth = 6378 # Km
mu_earth = 3.986004418e9