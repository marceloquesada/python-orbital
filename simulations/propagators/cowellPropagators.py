import numpy as np
from scipy.integrate import solve_ivp
from orbital_elements.oeOps import (
    get_orbital_elements,
    get_period
)
from utils.types import OrbitalElements
from utils.visualization import (
    plot_classic_orbital_elements,
    plot_3D_view
)
from tqdm import tqdm


class TwoBodyPropagator:
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

    def propagate(self, t: np.typing.NDArray, periods: float = None, integration_method: str = 'RK45', max_step: int = 5) -> np.typing.NDArray:
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
            max_step=max_step
        )

        X = solution.y

        state_vectors = X
        self.state_vectors = state_vectors

        return t, state_vectors


    def to_orbital_elements(self) -> list[OrbitalElements]:
        orbital_elementss = []
        for i in range(self.state_vectors.shape[1]):
            orbital_elementss.append(get_orbital_elements(self.state_vectors[:, i], self.mu))
        self.orbital_elementss = orbital_elementss

        return orbital_elementss


class PerturbedPropagator:
    def __init__(self, state_vector_0, mu: float, perturbation_funcs: list[any]):
        self.state_vector_0 = state_vector_0
        self.mu = mu
        self.perturbations = perturbation_funcs

    def _x_dot(self, t, x):
        xdot = np.zeros_like(x)
        xdot[0:3] = x[3:6]
        a = -(self.mu/(np.linalg.norm(x[0:3]))**3)*x[0:3]
        
        dt = t - self.t_1
        for perturbation in self.perturbations:
            a += perturbation.get_acceleration(x, dt)

        xdot[3:6] = a

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

        self.t_1 = t[0]
        solution = solve_ivp(self._x_dot, (t[0], t[-1]), self.state_vector_0, t_eval=t, method=integration_method)
        X = solution.y

        state_vectors = X
        self.state_vectors = state_vectors

        return t, state_vectors

    def to_orbital_elements(self) -> list[OrbitalElements]:
        orbital_elementss = []
        for i in range(self.state_vectors.shape[1]):
            orbital_elementss.append(get_orbital_elements(self.state_vectors[:, i], self.mu))
        self.orbital_elementss = orbital_elementss

        return orbital_elementss