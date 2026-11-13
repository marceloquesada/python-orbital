import numpy as np
from scipy.integrate import solve_ivp
from orbital_elements.elements import *
from utils.visualization import (
    plot_classic_orbital_elements,
    plot_3D_view
)
from orbital_utils import constants
from tqdm import tqdm
from perturbations import atmospheric_drag

class Unperturbed_propagator:
    """
    Propagates the motion of a body under the influence of a central gravitational field using the two-body problem formulation.

    Attributes
    ----------
    state_vectors : list
        Stores the propagated state vectors (position and velocity) at each time step.
    orbital_elements : list
        Stores the computed orbital elements at each time step.

    Parameters
    ----------
    state_vector_0 : array-like
        Initial state vector [x, y, z, vx, vy, vz] of the body.
    mu : float
        Standard gravitational parameter of the central body.

    Methods
    -------
    propagate(t, periods=None, integration_method='RK45')
        Propagates the state vector over the specified time array and number of periods using a chosen integration method.
    to_orbital_elements()
        Converts the propagated state vectors to orbital elements for each time step.

    Raises
    ------
    Exception
        If the provided time array cannot cover the specified number of periods.
    """

    def __init__(self, state_vector_0, mu: float):
        self.state_vector_0 = state_vector_0
        self.mu = mu

    def _x_dot(self, t, x):
        xdot = np.zeros_like(x)
        xdot[0:3] = x[3:6]
        xdot[3:6] = -(self.mu/(np.linalg.norm(x[0:3]))**3)*x[0:3]


        self.progress_bar.update(round(t -self.first_t -self.current_t))
        self.current_t = t

        return xdot

    def propagate(self, t: np.typing.NDArray, periods: float = None, integration_method: str = 'RK45',
                  **solver_options) -> np.typing.NDArray:
        """
        solver_options are forwarded to scipy's solve_ivp (e.g. rtol, atol, max_step).
        max_step defaults to the mean spacing of t.
        """
        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        if periods is not None:
            T = get_period(r, v, self.mu)

            total_t = T*periods

            if total_t < t[-1]:
                t = t[:np.where(t > total_t)[0][0]]
            else:
                raise Exception("The time array provided cannot cover the specified number of periods, either increase the array or decrease the number of periods")

        # Handle progress bar
        self.first_t = t[0]
        self.last_t = t[-1]
        self.current_t = t[0]
        
        self.progress_bar = tqdm(total=round(t[-1] - t[0] + 1))

        options = {"max_step": np.mean(np.diff(t))}
        options.update(solver_options)

        solution = solve_ivp(
            self._x_dot,
            (t[0], t[-1]),
            self.state_vector_0,
            t_eval=t,
            method=integration_method,
            **options
        )

        X = solution.y

        state_vectors = X
        self.state_vectors = state_vectors

        return t, state_vectors


    def to_orbital_elements(self) -> np.typing.NDArray:
        orbital_elementss = np.zeros((6, self.state_vectors.shape[1]))
        for i in range(self.state_vectors.shape[1]):
            orbital_elementss[:, i] = get_osculating_elements(self.state_vectors[:, i], self.mu)
        self.orbital_elementss = orbital_elementss

        return orbital_elementss


class Perturbed_propagator:
    def __init__(self, state_vector_0, mu: float, perturbation_funcs: list[any]):
        self.state_vector_0 = state_vector_0
        self.mu = mu
        self.perturbations = perturbation_funcs

    def _x_dot(self, t, x):
        # if get_major_axis(x[0:3], x[3:6], self.mu) < (constants.radius_earth + 5):  # Avoid crash
        #     return np.zeros_like(x)
        xdot = np.zeros_like(x)
        xdot[0:3] = x[3:6]
        a = -(self.mu/(np.linalg.norm(x[0:3]))**3)*x[0:3]
        
        for perturbation in self.perturbations:
            a += perturbation.get_acceleration(x, t)

        xdot[3:6] = a

        self.progress_bar.update(round(t-self.current_t))
        self.current_t = t

        return xdot

    def propagate(self, t: np.typing.NDArray, periods: float = None, integration_method: str = 'RK45',
                  **solver_options) -> np.typing.NDArray:
        """
        solver_options are forwarded to scipy's solve_ivp (e.g. events, rtol, atol, max_step).
        max_step defaults to the largest spacing of t. If a terminal event fires, the returned
        time array and state vectors stop at the last sample before the event; the full
        solve_ivp result (including t_events) is kept in self.solution.
        """
        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        # Handle progress bar
        self.first_t = t[0]
        print(t[0])
        self.last_t = t[-1]
        self.current_t = t[0]

        self.progress_bar = tqdm(total=round(t[-1] - t[0] + 1))

        options = {"max_step": np.max(np.diff(t))}
        options.update(solver_options)

        try:
            solution = solve_ivp(
                        self._x_dot,
                        (t[0], t[-1]),
                        self.state_vector_0,
                        t_eval=t,
                        method=integration_method,
                        **options
                    )
        finally:
            # Close explicitly: an unclosed tqdm is torn down during interpreter shutdown,
            # which segfaults when a Qt matplotlib backend is loaded
            self.progress_bar.close()
        self.solution = solution
        X = solution.y

        state_vectors = X
        self.state_vectors = state_vectors

        return solution.t, state_vectors

    def to_orbital_elements(self) -> np.typing.NDArray:
        orbital_elementss = np.zeros((6, self.state_vectors.shape[1]))

        for i in range(self.state_vectors.shape[1]):
            orbital_elements = get_osculating_elements(self.state_vectors[:, i], self.mu)
            orbital_elementss[:, i] = orbital_elements
        
        return orbital_elementss

class Drag_propagator:
    def __init__(self, state_vector_0, mu: float, mass, area, coef_drag):
        self.state_vector_0 = state_vector_0
        self.mu = mu
        self.mass = mass
        self.area = area
        self.coef_drag = coef_drag

    def _x_dot(self, t, x):
        # if get_major_axis(x[0:3], x[3:6], self.mu) < (constants.radius_earth + 5):  # Avoid crash
        #     return np.zeros_like(x)
        xdot = np.zeros_like(x)
        xdot[0:3] = x[3:6]
        a = -(self.mu/(np.linalg.norm(x[0:3]))**3)*x[0:3]
        
        dt = t - self.first_t
        
        a += atmospheric_drag.compiled_get_acceleration(x, self.coef_drag, self.area, self.mass)

        xdot[3:6] = a

        self.progress_bar.update(round(t -self.first_t -self.current_t))
        self.current_t = t

        return xdot

    def propagate(self, t: np.typing.NDArray, periods: float = None, integration_method: str = 'RK45') -> np.typing.NDArray:
        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]
        if periods is not None:
            T = get_period(r, v, self.mu)

            total_t = T*periods

            if total_t < t[-1]:
                t = t[:np.where(t > total_t)[0][0]]
            else:
                raise Exception("The time array provided cannot cover the specified number of periods, either increase the array or decrease the number of periods")
        

         # Handle progress bar
        self.first_t = t[0]
        self.last_t = t[-1]
        self.current_t = t[0]
        
        self.progress_bar = tqdm(total=round(t[-1] - t[0] + 1))

        solution = solve_ivp(
                    self._x_dot,
                    (t[0], t[-1]),
                    self.state_vector_0,
                    t_eval=t,
                    method=integration_method,
                    max_step=np.mean(np.diff(t))
                )
        X = solution.y

        state_vectors = X
        self.state_vectors = state_vectors

        return t, state_vectors

    def to_orbital_elements(self) -> np.typing.NDArray:
        orbital_elementss = np.zeros((6, self.state_vectors.shape[1]))

        for i in range(self.state_vectors.shape[1]):
            orbital_elements = get_osculating_elements(self.state_vectors[:, i], self.mu)
            orbital_elementss[:, i] = orbital_elements
        
        return orbital_elementss