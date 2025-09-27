import numpy as np
from utils.types import OrbitalElements
from utils import orbitalElementsOperations
from utils import refSystems
from .constants import Bases


def get_state_vectors(orbital_elements: OrbitalElements, mu: float) -> np.typing.NDArray:
    a = orbital_elements.major_axis
    e = orbital_elements.eccentricity
    theta = orbital_elements.true_anomaly

    theta_rad = np.deg2rad(theta)

    p = (a*(1 - e**2))
    r = p/(1 + e*np.cos(theta_rad))

    r_p = np.array([r * np.cos(theta_rad), r * np.sin(theta_rad), 0])
    v_p = np.sqrt(mu/p)*np.array([-np.sin(theta_rad), e + np.cos(theta_rad), 0])

    r_I = refSystems.perifocal_to_inertial(r_p, orbital_elements)
    v_I = refSystems.perifocal_to_inertial(v_p, orbital_elements)

    X_I = np.concatenate((r_I, v_I))


    return X_I

# OLD - Does NOT work
# def get_state_vectors(orbital_elements: OrbitalElements, mu: float) -> np.typing.NDArray:
#     a = orbital_elements.major_axis
#     e = orbital_elements.eccentricity
#     i = orbital_elements.inclination
#     Omega = orbital_elements.ascending_node
#     omega = orbital_elements.argument_of_perigee
#     theta = orbital_elements.true_anomaly

#     theta_rad = np.deg2rad(theta)

#     E = orbitalElementsOperations.get_eccentric_anomaly(theta_rad, e)
#     M = orbitalElementsOperations.get_mean_anomaly(theta_rad, e, mu)
#     print(M)

#     r = a*(1 - e*np.cos(E))

#     r_p = np.array([r * np.cos(theta_rad), r * np.sin(theta_rad), 0])
#     v_p = (np.sqrt(mu*a)/r)*np.array([-np.sin(E), np.sqrt(1 - e**2)*np.cos(E), 0])

#     r_I = refSystems.perifocal_to_inertial(r_p, orbital_elements)
#     v_I = refSystems.perifocal_to_inertial(v_p, orbital_elements)

#     X_I = np.concatenate((r_I, v_I))


#     return X_I