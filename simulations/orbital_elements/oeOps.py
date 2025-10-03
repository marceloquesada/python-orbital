import numpy as np
from utils.types import OrbitalElements
from utils.constants import Bases


MIN_ALLOWED_VALUE = 1e-10


## SINGLE VALUE FUNCTIONS
def get_orbital_elements(X: np.typing.NDArray, mu: float) -> OrbitalElements:
    r = X[0:3]
    v = X[3:6]

    e = get_eccentricity(r, v, mu)
    i = get_inclination(r, v, mu)

    if e < 0:
        raise Exception("Orbit is parabolic, only elliptical orbits are supported")
    elif e > 1:
        raise Exception("Orbit is hyperbolic, only elliptical orbits are supported")
    
    if e == 0:
        orbit_type = "circular"
    else:
        orbit_type = "elliptical"

    if i == 0:
        orbit_type += ", equatorial"

    if "elliptical" in orbit_type:
        if "equatorial" in orbit_type:
            orbital_elements = OrbitalElements(
                major_axis=get_major_axis(r, v, mu),
                eccentricity=get_eccentricity(r, v, mu),
                inclination=get_inclination(r, v, mu),
                longitude_of_periapsis=get_longitude_of_periapsis(r, v, mu),
                true_anomaly=get_true_anomaly(r, v, mu)
            )
        else:
            orbital_elements = OrbitalElements(
                major_axis=get_major_axis(r, v, mu),
                eccentricity=get_eccentricity(r, v, mu),
                inclination=get_inclination(r, v, mu),
                ascending_node=get_ascending_node(r, v, mu),
                argument_of_perigee=get_argument_of_perigee(r, v, mu),
                true_anomaly=get_true_anomaly(r, v, mu)
            )
    elif orbit_type == "circular":
        if "equatorial" in orbit_type:
            orbital_elements = OrbitalElements(
                major_axis=get_major_axis(r, v, mu),
                inclination=get_inclination(r, v, mu),
                ascending_node=get_ascending_node(r, v, mu),
                argument_of_latitude=get_argument_of_latitude(r, v, mu)
            )
        else:
            orbital_elements = OrbitalElements(
                major_axis=get_major_axis(r, v, mu),
                longitude_of_periapsis=get_longitude_of_periapsis(r, v, mu),
                argument_of_latitude=get_argument_of_latitude(r, v, mu)
            )
    
    orbital_elements.orbit_type = orbit_type
    

    return orbital_elements


def get_major_axis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    eps = (v_norm**2)/2 - (mu/r_norm)
    a = -mu/(2*eps)

    return a


def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)
    e_norm = np.linalg.norm(e)

    if abs(e_norm) < MIN_ALLOWED_VALUE:
        e_norm = 0

    return e_norm

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    # atan2(sin(i), cos(i)), where sin(i) = sqrt(hx^2 + hy^2)/|h|, cos(i) = hz/|h|
    sin_i = np.sqrt(h[0]**2 + h[1]**2) / h_norm
    cos_i = h[2] / h_norm

    i_rad = np.arctan2(sin_i, cos_i)

    i = np.rad2deg(i_rad)

    if i < MIN_ALLOWED_VALUE:
        i = 0

    return i


def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    if np.linalg.norm(n) > 1e-10:
        Omega_rad = np.acos(np.dot(Bases.i, n)/np.linalg.norm(n))
        if n[1] < 0:
            Omega_rad = 2*np.pi - Omega_rad
    else:
        Omega_rad = 0

    Omega = np.rad2deg(Omega_rad)

    return Omega


def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec)
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    if np.linalg.norm(n) > 1e-10 and e != 0:
        omega_rad = np.acos(np.dot(n, e_vec)/(np.linalg.norm(n)*e))
        if e_vec[2] < 0:
            omega_rad = 2*np.pi - omega_rad
    else:
        omega_rad = 0

    omega = np.rad2deg(omega_rad)

    return omega


def get_true_anomaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)

    dot_er = np.dot(e, r)
    cross_er = np.cross(e, r)
    theta_rad = np.atan2(np.linalg.norm(cross_er), dot_er)

    if np.dot(r, v) < 0:
        theta_rad = 2*np.pi - theta_rad
    if np.linalg.norm(e) < MIN_ALLOWED_VALUE:  # Handle circular orbits
        theta_rad = 0

    theta = np.rad2deg(theta_rad)

    return theta


def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)  # angular mommentum
    e = (np.cross(v, h)/mu) - (r/np.linalg.norm(r))

    if np.linalg.norm(e) < MIN_ALLOWED_VALUE:  # Handle circular orbits
        e = np.zeros_like(e)

    return e


def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    period = 2*np.pi*np.sqrt((a**3)/mu)

    return period


# ALTERNATIVE ELEMENTS
def get_argument_of_latitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    dot_rn = np.dot(r, n)
    cross_rn = np.cross(r, n)

    sigma_rad = np.atan2(np.linalg.norm(cross_rn), dot_rn)

    sigma_deg = np.rad2deg(sigma_rad)

    return sigma_deg


def get_longitude_of_periapsis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec)
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    if np.linalg.norm(n) > 1e-10 and e != 0:
        varpi_rad = np.acos(np.dot(Bases.i, e_vec)/(np.linalg.norm(Bases.i)*e))
        if e_vec[1] < 0:
            varpi_rad = 2*np.pi - varpi_rad
    else:
        varpi_rad = 0

    varpi = np.rad2deg(varpi_rad)

    return varpi

def get_true_longitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    dot_ri = np.dot(r, Bases.i)
    cross_ri = np.cross(r, Bases.i)

    l_rad = np.atan2(np.linalg.norm(cross_ri), dot_ri)

    l_deg = np.rad2deg(l_rad)

    return l_deg


# KEPLERIAN ELEMENTS
def get_eccentric_anomaly(theta_rad, e) -> float:
    E_sin = np.sqrt(1-e**2)*np.sin(theta_rad)
    E_cos = 1 + e*np.cos(theta_rad)

    E = np.atan2(E_sin, E_cos)

    return E


def get_mean_angular_motion(period, mu: float) -> float:
    n = 2*np.pi/period

    return n


def get_mean_anomaly(theta_rad, e, mu: float) -> float:
    E = get_eccentric_anomaly(theta_rad, e)
    M = E - e*np.sin(E)

    return M


def get_analitical_time(theta: float, e: float, period: float, t0: float, mu: float) -> float:
    M = get_mean_anomaly(theta, e, mu)
    n = get_mean_angular_motion(period, mu)

    t = t0 + ((M)/n)

    return t


def get_true_anomaly_from_mean(e: float | np.typing.NDArray, M: float | np.typing.NDArray, max_iterations: int = 30, tolerance: float = 1e-8) -> float | np.typing.NDArray:
    theta_rad = None

    if M < np.pi:
        E_i = M + (e/2)
    else:
        E_i = M - (e/2)

    for i in range(max_iterations):
        f = M - E_i + e*np.sin(E_i)
        df = -1 + e*np.cos(E_i)
        E_i1 = E_i - (f/df)
        if abs(E_i1 - E_i) <= tolerance:
            theta_rad = 2*np.atan(np.sqrt((1+e)/(1-e)) * np.tan(E_i1/2))
            break
        E_i = E_i1

    if theta_rad is None:
        raise Exception("Given number of iterations is not enough for achieving given tolerance")
    
    theta = np.rad2deg(theta_rad)
    
    return theta

