import numpy as np
from orbital_elements.elements import *
from orbital_utils import ref_systems
from copy import copy


class Two_body_analytical_propagator:
    def __init__(self, state_vector_0, mu: float):
        self.state_vector_0 = state_vector_0
        self.mu = mu

        self.orbital_elements = get_osculating_elements(state_vector_0, self.mu)

    def _get_timestamps(self):
        theta_0 = self.thetas_rad[0]
        period = get_period(self.state_vector_0[:, 0:3], self.state_vector_0[:, 3:6], self.mu)
        e = self.orbital_elements[1]
        mu = self.mu
        t_0 = get_analitical_time(theta_0, e, period, 0, mu)

        ts = np.array([])
        for theta in self.thetas_rad:
            t = get_analitical_time(theta, e, period, t_0, mu)
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
        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        theta_0 = get_true_anomaly(r, v, self.mu)
        thetas = np.arange(theta_0, (periods*360) + theta_0, step_size)

        thetas_rad = np.deg2rad(thetas)

        self.thetas = thetas
        self.thetas_rad = thetas_rad

        h = np.cross(r, v)
        e = self.orbital_elements[1]
        p = np.linalg.norm(h)**2/self.mu

        rs = p / (1 + e * np.cos(thetas_rad))

        X_p = np.array([rs * np.cos(thetas_rad), rs * np.sin(thetas_rad), np.zeros_like(thetas_rad)])

        ts = self.time_at_true_anomaly(r, v, self.mu, thetas_rad, t0=0)
        ts[0] = 0

        return ts, X_p

    def propagate(self, periods: int = 1, step_size: float = 0.01):
        ts, X_p = self.propagate_2d(periods, step_size)
        X_I = ref_systems.perifocal_to_inertial(X_p, self.orbital_elements)

        return ts, X_I

    def to_orbital_elements(self):
        orbital_elementss = np.zeros((6, len(self.thetas)))

        wrapped_thetas = self.thetas % 360

        for i in range(len(wrapped_thetas)):
            theta = wrapped_thetas[i]
            orbital_elements = copy(self.orbital_elements)
            orbital_elements[5] = theta
            orbital_elementss[:, i] = orbital_elements

        self.orbital_elementss = orbital_elementss

        return orbital_elementss


class Continuous_two_body_analytical_propagator:
    """
    Closed-form (Keplerian) two-body propagator that returns the state vector
    at any time t (seconds since the epoch of state_vector_0).

    Usage:
        prop = Continuous_two_body_analytical_propagator(state_vector_0, mu)
        X = prop(t)              # scalar t -> shape (6,)
        Xs = prop(np.array(ts))  # array ts -> shape (6, N)

    Kepler's equation M = E - e*sin(E) is solved with Newton-Raphson, vectorized
    over all requested times. Only elliptical orbits (0 <= e < 1) are supported.
    """

    def __init__(self, state_vector_0, mu: float, tolerance: float = 1e-12, max_iterations: int = 50):
        self.state_vector_0 = np.asarray(state_vector_0, dtype=float)
        self.mu = mu
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        r0 = self.state_vector_0[0:3]
        v0 = self.state_vector_0[3:6]
        r0_norm = np.linalg.norm(r0)

        h_vec = np.cross(r0, v0)
        self.h = np.linalg.norm(h_vec)
        e_vec = np.cross(v0, h_vec) / mu - r0 / r0_norm
        self.e = np.linalg.norm(e_vec)

        if self.e >= 1.0:
            raise ValueError("Only elliptical orbits (e < 1) are supported")

        self.a = 1 / (2 / r0_norm - np.dot(v0, v0) / mu)
        self.p = self.h**2 / mu
        self.n = np.sqrt(mu / self.a**3)
        self.period = 2 * np.pi / self.n

        # Perifocal basis built directly from the vectors (robust for circular
        # and/or equatorial orbits, where Omega/omega are undefined).
        W = h_vec / self.h
        if self.e > 1e-10:
            P = e_vec / self.e
        else:
            P = r0 / r0_norm  # circular: measure anomaly from the initial position
        Q = np.cross(W, P)
        self.R_pqw_to_inertial = np.column_stack((P, Q, W))

        # Initial true, eccentric and mean anomalies
        theta0 = np.arctan2(np.dot(r0, Q), np.dot(r0, P))
        E0 = self._true_to_eccentric(theta0)
        self.M0 = E0 - self.e * np.sin(E0)

        self.orbital_elements = get_osculating_elements(self.state_vector_0, mu)

    def _true_to_eccentric(self, theta):
        e = self.e
        return np.arctan2(np.sqrt(1 - e**2) * np.sin(theta), e + np.cos(theta))

    def _solve_kepler(self, M):
        e = self.e
        M = np.mod(M, 2 * np.pi)
        E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
        for _ in range(self.max_iterations):
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E = E - dE
            if np.all(np.abs(dE) < self.tolerance):
                break
        return E

    def true_anomaly(self, t):
        """True anomaly [rad] at time(s) t [s]."""
        M = self.M0 + self.n * np.asarray(t, dtype=float)
        E = self._solve_kepler(M)
        e = self.e
        return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

    def state_at(self, t):
        """
        State vector [x, y, z, vx, vy, vz] in the inertial frame at time(s) t [s].
        Scalar t -> shape (6,); array t -> shape (6, N).
        """
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)

        theta = self.true_anomaly(t_arr)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r = self.p / (1 + self.e * cos_t)

        zeros = np.zeros_like(theta)
        r_pqw = np.array([r * cos_t, r * sin_t, zeros])
        v_pqw = np.sqrt(self.mu / self.p) * np.array([-sin_t, self.e + cos_t, zeros])

        X = np.vstack((self.R_pqw_to_inertial @ r_pqw, self.R_pqw_to_inertial @ v_pqw))

        return X[:, 0] if scalar else X

    def __call__(self, t):
        return self.state_at(t)

    def propagate(self, t_final: float, step_size: float, t_initial: float = 0.0):
        """Sample the orbit on a uniform time grid. Returns (ts, X) with X of shape (6, N)."""
        ts = np.arange(t_initial, t_final + step_size / 2, step_size)
        return ts, self.state_at(ts)

    def orbital_elements_at(self, t):
        """Orbital elements [a, e, i, Omega, omega, theta] (angles in degrees) at time(s) t."""
        theta_deg = np.rad2deg(np.mod(self.true_anomaly(t), 2 * np.pi))
        if np.ndim(theta_deg) == 0:
            oe = copy(self.orbital_elements)
            oe[5] = theta_deg
            return oe
        oe = np.tile(self.orbital_elements.reshape(6, 1), (1, len(theta_deg)))
        oe[5, :] = theta_deg
        return oe
