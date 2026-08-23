import numpy as np
from ..orbital_utils import ref_systems


def get_state_vectors(oe: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    a = oe[:, 0]
    e = oe[:, 1]
    theta = oe[:, 2]

    theta_rad = np.deg2rad(theta)

    p = (a*(1 - e**2))
    r = p/(1 + e*np.cos(theta_rad))

    r_p = np.array([r * np.cos(theta_rad), r * np.sin(theta_rad), 0])
    v_p = np.sqrt(mu/p)*np.array([-np.sin(theta_rad), e + np.cos(theta_rad), 0])

    r_I = ref_systems.perifocal_to_inertial(r_p, oe)
    v_I = ref_systems.perifocal_to_inertial(v_p, oe)

    X_I = np.transpose(np.concatenate((r_I, v_I)))

    return X_I
