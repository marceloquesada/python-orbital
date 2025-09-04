import numpy as np
from scipy.integrate import solve_ivp
from utils.orbitalElements import (
    get_orbital_elements,
    get_period
)
from utils.types import OrbitalElements
from utils.visualization import (
    plot_classic_orbital_elements,
    plot_3D_view
)


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


# Essa classe me dá mais desgosto que a minha vida, ta tudo desorganizado e as perturbações não funcionam direito
class DisturbedPropagator:
    state_vectors = []
    orbital_elements = []

    def __init__(
            self,
            state_vector_0,
            mu: float,
            use_thust: bool = False,
            use_drag: bool = False,
            use_J2: bool = False
            ):
        self.state_vector_0 = state_vector_0
        self.mu = mu

        self.use_thust = use_thust
        self.use_drag = use_drag
        self.use_J2 = use_J2

    def _get_bases(self, x):
        r = x[0:3]
        v = x[3:6]

        bases = np.array([np.linalg.norm(r), np.linalg.norm(v), np.linalg.norm(np.cross(r, v))])

        return bases

    def _thrust_perturbation(self, bases, thrust_vector):

        pass

    def _velocity_function(self, t, x, **kwargs):
        bases = self._get_bases(x)

        xdot = np.zeros_like(x)
        xdot[0:3] = x[3:6]
        total_acceleration = 0

        # Two body acceleration
        total_acceleration += -(self.mu/(np.linalg.norm(x[0:3]))**3)*x[0:3]

        # Thrust perturbation
        if self.use_thust:
            thrust_vector = kwargs.get('thrust_vector', np.zeros(3))
            thrust_acceleration = self._thrust_perturbation(bases, thrust_vector)  # Eu quero me matar toda vez que leio isso
            total_acceleration += thrust_acceleration

        xdot[3:6] = total_acceleration

        return xdot

    def propagate(
            self,
            t: np.typing.NDArray,
            periods: float = None,
            integration_method: str = 'RK45',
            **kwargs) -> np.typing.NDArray:

        # Decompose state vector to r and v vectors
        r = self.state_vector_0[0:3]
        v = self.state_vector_0[3:6]

        # If periods is not None, calculate the total period of 'periods' orbits and cutoff time array t accordingly
        if periods is not None:
            T = get_period(r, v, self.mu)
        total_t = T*periods
        if total_t < t[-1]:
            t = t[:np.where(t > total_t)[0][0]]
        else:
            raise Exception("The time array provided cannot cover the specified number of periods, either increase the array or decrease the number of periods")

        def ode_function(t, x):
            return self._velocity_function(t, x, **kwargs)

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
