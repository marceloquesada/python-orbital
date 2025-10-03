import numpy as np
from orbital_elements import oeOps
from utils import refSystems
from copy import copy


class TwoBodyAnalyticalPropagator:
    def __init__(self, state_vector_0, mu: float):
        self.state_vector_0 = state_vector_0
        self.mu = mu

        self.orbital_elements = oeOps.get_orbital_elements(state_vector_0, self.mu)

    def _get_timestamps(self):
        theta_0 = self.thetas_rad[0]
        period = oeOps.get_period(self.state_vector_0[0:3], self.state_vector_0[3:6], self.mu)
        e = self.orbital_elements.eccentricity
        mu = self.mu
        t_0 = oeOps.get_analitical_time(theta_0, e, period, 0, mu)

        ts = np.array([])
        for theta in self.thetas_rad:
            t = oeOps.get_analitical_time(theta, e, period, t_0, mu)
            ts = np.concatenate((ts, np.array([t])))

        return ts

    def time_at_true_anomaly(self, r0, v0, mu, thetas, t0=0):
        r_norm = np.linalg.norm(r0)
        v_norm = np.linalg.norm(v0)
        
        h_vec = np.cross(r0, v0)
        h = np.linalg.norm(h_vec)
        
        e_vec = ((v_norm**2 - mu/r_norm) * r0 - np.dot(r0, v0) * v0) / mu
        e = np.linalg.norm(e_vec)
        
        a = 1 / (2/r_norm - v_norm**2/mu)
        
        if e < 1e-8:
            theta0 = 0.0
        else:
            p = h**2 / mu
            cos_theta0 = (p / r_norm - 1) / e
            sin_theta0 = np.dot(r0, v0) * h / (mu * e * r_norm)
            theta0 = np.arctan2(sin_theta0, cos_theta0)
            theta0 = theta0 % (2 * np.pi)
        
        n = np.sqrt(mu / np.abs(a)**3)
        if e < 1.0:
            E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta0 / 2))
            M0 = E0 - e * np.sin(E0)
            t_peri0 = M0 / n
        else:
            raise ValueError("Hyperbolic orbits not supported in this example")
        
        times = []
        for theta in thetas:
            theta = theta % (2 * np.pi)
            if e < 1.0:
                E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta / 2))
                M = E - e * np.sin(E)
                t_peri = M / n
            else:
                raise ValueError("Hyperbolic orbits not supported in this example")
            
            dt = t_peri - t_peri0
            if dt < 0:
                dt += 2 * np.pi / n
            times.append(t0 + dt)
    
        return times

    def propagate_2d(self, periods: int = 1, step_size: float = 0.01):
        theta_0 = oeOps.get_true_anomaly(self.state_vector_0[0:3], self.state_vector_0[3:6], self.mu)
        thetas = np.arange(theta_0, (periods*360) + theta_0, step_size)

        thetas_rad = np.deg2rad(thetas)

        self.thetas = thetas
        self.thetas_rad = thetas_rad

        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        h = np.cross(r, v)
        e = self.orbital_elements.eccentricity
        p = np.linalg.norm(h)**2/self.mu

        rs = p / (1 + e * np.cos(thetas_rad))

        X_p = np.array([rs * np.cos(thetas_rad), rs * np.sin(thetas_rad), np.zeros_like(thetas_rad)])

        ts = self.time_at_true_anomaly(r, v, self.mu, thetas_rad, t0=0)

        return ts, X_p

    def propagate(self, periods: int = 1, step_size: float = 0.01):
        ts, X_p = self.propagate_2d(periods, step_size)
        X_I = refSystems.perifocal_to_inertial(X_p, self.orbital_elements)

        return ts, X_I

    def to_orbital_elements(self):
        orbital_elementss = []

        wrapped_thetas = self.thetas % 360

        for theta in wrapped_thetas:
            orbital_elements = copy(self.orbital_elements)
            orbital_elements.true_anomaly = theta
            orbital_elementss.append(orbital_elements)

        self.orbital_elementss = orbital_elementss

        return orbital_elementss
