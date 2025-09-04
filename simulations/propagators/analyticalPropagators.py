import numpy as np
from utils import refSystems, orbitalElements


class TwoBodyAnalyticalPropagator:
    def __init__(self, state_vector_0, mu: float):
        self.state_vector_0 = state_vector_0
        self.mu = mu

        self.orbital_elements = orbitalElements.get_orbital_elements(state_vector_0, self.mu)

    def propagate_2d(self, periods: int = 1, step_size: float = 0.01):
        thetas = np.arange(0, periods*360, step_size)
        thetas_rad = np.deg2rad(thetas)

        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        h = np.cross(r, v)
        e = self.orbital_elements.eccentricity
        p = np.linalg.norm(h)**2/self.mu

        rs = p / (1 + e * np.cos(thetas_rad))

        X_p = np.array([rs * np.cos(thetas_rad), rs * np.sin(thetas_rad), np.zeros_like(thetas_rad)])

        return X_p

    def propagate(self, periods: int = 1, step_size: float = 0.01):
        X_p = self.propagate_2d(periods, step_size)
        X_I = refSystems.perifocal_to_inertial(X_p, self.orbital_elements)

        return X_I
