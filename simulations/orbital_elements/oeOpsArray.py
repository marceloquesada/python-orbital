import numpy as np
from utils.types import OrbitalElements
from utils.constants import Bases
import warnings


MIN_ALLOWED_VALUE = 1e-10


## ARRAY FUNCTIONS
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


def get_major_axis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # OK
    r_norm = np.linalg.norm(r, axis=0)
    v_norm = np.linalg.norm(v, axis=0)
    eps = (v_norm**2)/2 - (mu/r_norm)
    a = -mu/(2*eps)

    return a


def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # OK
    e = get_eccentricity_vector(r, v, mu)
    e_norm = np.linalg.norm(e, axis=0)

    e_norm[e_norm < MIN_ALLOWED_VALUE] = 0

    return e_norm

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray:# OK
    h = np.cross(r, v, axis=0)
    h_norm = np.linalg.norm(h, axis=0)

    sin_i = np.sqrt(h[0, :]**2 + h[1, :]**2) / h_norm
    cos_i = h[2, :] / h_norm

    i_rad = np.atan2(sin_i, cos_i)

    i = np.rad2deg(i_rad)

    i[i < MIN_ALLOWED_VALUE] = 0

    return i


def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:# OK
    h = np.cross(r, v, axis=0)
    n = np.cross(Bases.k, h, axis=0)
    n_norm = np.linalg.norm(n, axis=0)

    Omega_rad =  np.acos(np.dot(Bases.i, n)/np.linalg.norm(n, axis=0))
    Omega_rad[n_norm < MIN_ALLOWED_VALUE] = 0  # Deal with very small values

    Omega_rad[n[1, :] < 0] =  2*np.pi - Omega_rad[n[1, :] < 0]

    Omega = np.rad2deg(Omega_rad)

    return Omega


def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray:# OK
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec, axis=0)
    h = np.cross(r, v, axis=0)
    n = np.cross(Bases.k, h, axis=0)
    n_norm = np.linalg.norm(n, axis=0)

    omega_rad =  np.arccos(np.sum(n * e_vec, axis=0) / (np.linalg.norm(n, axis=0) * e))

    omega_rad[n_norm < MIN_ALLOWED_VALUE] = 0  # Deal with very small values
    omega_rad[e == 0] = 0 # Deal with circular orbits

    omega_rad[n[2, :] < 0] = 2*np.pi - omega_rad[n[2, :] < 0]

    omega = np.rad2deg(omega_rad)

    return omega


def get_true_anomaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray:# OK
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec, axis=0)

    dot_er = np.sum(e_vec*r, axis=0)
    cross_er = np.cross(e_vec, r, axis=0)
    theta_rad = np.arctan2(np.linalg.norm(cross_er, axis=0), dot_er)

    theta_rad[e == 0] = 0  # Deal with circular orbits

    print(r.shape, v.shape)
    mask = np.sum(r*v, axis=0) < 0
    theta_rad[mask] = 2*np.pi - theta_rad[mask]

    theta = np.rad2deg(theta_rad)

    return theta


def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray: # OK
    h = np.cross(r, v, axis=0)  # angular mommentum
    e = (np.cross(v, h, axis=0)/mu) - (r/np.linalg.norm(r, axis=0))

    return e


def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # OK
    a = get_major_axis(r, v, mu)
    period = 2*np.pi*np.sqrt((a**3)/mu)

    return period


# ALTERNATIVE ELEMENTS
def get_argument_of_latitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # NOT WORKING -- FIX IMMEDIATLY
    h = np.cross(r, v, axis=0)
    n = np.cross(Bases.k, h, axis=0)

    dot_rn = np.sum(r*n, axis=0)
    cross_rn = np.cross(r, n, axis=0)

    sigma_rad = np.atan2(np.linalg.norm(cross_rn, axis=0), dot_rn)
    
    mask = r[0, :] < 0  # Check if r_x < 0, if so, compensate so that 0 <= sigma < 360 
    sigma_rad[mask] = 2*np.pi - sigma_rad[mask]

    sigma = np.rad2deg(sigma_rad)

    return sigma

def get_argument_of_latitude2(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # NOT WORKING -- FIX IMMEDIATLY
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    dot_rn = np.dot(r, n)
    cross_rn = np.cross(r, n)

    sigma_rad = np.atan2(np.linalg.norm(cross_rn), dot_rn)
    
    if r[0] < 0:
        sigma_rad = 2*np.pi - sigma_rad

    sigma = np.rad2deg(sigma_rad)

    return sigma


def get_longitude_of_periapsis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float | np.typing.NDArray: # Maybe working? IDK
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec, axis=0)
    h = np.cross(r, v, axis=0)
    n = np.cross(Bases.k, h, axis=0)

    varpi_rad = np.acos(np.sum(np.transpose(np.tile(Bases.i, (e_vec.shape[1], 1)))*e_vec, axis=0)/(np.linalg.norm(Bases.i)*e))

    varpi_rad[e == 0] = 0
    varpi_rad[np.linalg.norm(n, axis=0) > 1e-10] = 0

    varpi_rad[e_vec[1, :] < 0] = 2*np.pi - varpi_rad[e_vec[1, :] < 0]

    varpi = np.rad2deg(varpi_rad)

    return varpi

def get_true_longitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float: # not done
    dot_ri = np.dot(r, Bases.i)
    cross_ri = np.cross(r, Bases.i)

    l_rad = np.atan2(np.linalg.norm(cross_ri), dot_ri)

    l_deg = np.rad2deg(l_rad)

    return l_deg


# KEPLERIAN ELEMENTS
def get_eccentric_anomaly(theta_rad, e) -> float: # not done
    E_sin = np.sqrt(1-e**2)*np.sin(theta_rad)
    E_cos = 1 + e*np.cos(theta_rad)

    E = np.atan2(E_sin, E_cos)

    return E


def get_mean_angular_motion(period, mu: float) -> float: # not done
    n = 2*np.pi/period

    return n


def get_mean_anomaly(theta_rad, e, mu: float) -> float: # not done
    E = get_eccentric_anomaly(theta_rad, e)
    M = E - e*np.sin(E)

    return M


def get_analitical_time(theta: float, e: float, period: float, t0: float, mu: float) -> float: # not done
    M = get_mean_anomaly(theta, e, mu)
    n = get_mean_angular_motion(period, mu)

    t = t0 + ((M)/n)

    return t


def get_true_anomaly_from_mean(e: np.typing.NDArray, M: np.typing.NDArray, max_iterations: int = 30, tolerance: float = 1e-8) -> np.typing.NDArray:
    theta_rad = None
    theta = np.empty_like(M)

    
    for row in range(M.shape[0]):
        Mj = M[row]

        if Mj < np.pi:
            E_i = Mj + (e/2)
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
            raise warnings.warn("Given number of iterations is not enough for achieving given tolerance - defaulting to None")
        
    theta[row] = np.rad2deg(theta_rad)
    
    return theta


    
        

    