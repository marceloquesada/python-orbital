import numpy as np
from utils.orbitalElements import OrbitalElements


def perifocal_to_inertial(points_perifocal: np.typing.NDArray, orbital_elements: OrbitalElements):
    omega = orbital_elements.argument_of_perigee
    Omega = orbital_elements.ascending_node
    i = orbital_elements.inclination

    rotation_matrix = np.array(
        [
            np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(i),
            -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(i),
            np.sin(Omega)*np.sin(i)
        ],
        [
            np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(i),
            -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(i),
            -np.cos(Omega)*np.sin(i)
        ],
        [
            np.sin(omega)*np.sin(i),
            np.cos(omega)*np.sin(i),
            np.cos(i)
        ]
    )

    points_inertial = np.dot(points_perifocal, rotation_matrix)

    return points_inertial
