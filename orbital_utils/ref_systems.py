import numpy as np


def get_base_versors(state_vector):
    r_hat = np.linalg.norm(state_vector[0:3])
    v_hat = np.linalg.norm(state_vector[3:6])
    h_hat = np.linalg.norm(np.cross(state_vector[0:3], state_vector[3:6]))

    base = np.array([r_hat, v_hat, h_hat])

    return base


# FUNDAMENTAL ROTATIONS
def rot_X(angle_deg: float, clockwise: bool = False) -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    if clockwise:
        Rx = Rx.transpose()
    else:
        Rx = Rx

    return Rx


def rot_Y(angle_deg: float, clockwise: bool = False) -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Ry = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    if clockwise:
        Ry = Ry.transpose()
    else:
        Ry = Ry

    return Ry


def rot_Z(angle_deg: float, clockwise: bool = False) -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Rz = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    if clockwise:
        Rz = Rz.transpose()
    else:
        Rz = Rz

    return Rz


# SYSTEM REFERENCE ROTATIONS
def perifocal_to_inertial(points_perifocal: np.typing.NDArray, orbital_elements: np.typing.NDArray):
    omega = orbital_elements[4]
    Omega = orbital_elements[3]
    i = orbital_elements[2]

    Rz_omega = rot_Z(omega)
    Rx_i = rot_X(i)
    Rz_Omega = rot_Z(Omega)

    rotation_matrix = Rz_Omega @ Rx_i @ Rz_omega

    points_inertial = rotation_matrix @ points_perifocal  # @ é equivalente ao produto matricial

    return points_inertial


def inertial_to_perifocal(points_inertial: np.typing.NDArray, orbital_elements: np.typing.NDArray):
    omega = orbital_elements[4]
    Omega = orbital_elements[3]
    i = orbital_elements[2]

    Rz_omega = rot_Z(omega, True)
    Rx_i = rot_X(i, True)
    Rz_Omega = rot_Z(Omega, True)

    rotation_matrix = Rz_omega @ Rx_i @ Rz_Omega

    points_perifocal = rotation_matrix @ points_inertial

    return points_perifocal


def orbital_to_inertial(points_orbital: np.typing.NDArray, orbital_elements: np.typing.NDArray):
    omega = orbital_elements[4]
    Omega = orbital_elements[3]
    i = orbital_elements[2]
    theta = orbital_elements[5]

    Rz_omega = rot_Z(omega+theta)
    Rx_i = rot_X(i)
    Rz_Omega = rot_Z(Omega)

    rotation_matrix = Rz_Omega @ Rx_i @ Rz_omega

    points_inertial = rotation_matrix @ points_orbital  # @ é equivalente ao produto matricial

    return points_inertial
