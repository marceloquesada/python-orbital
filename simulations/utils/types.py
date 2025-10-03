from dataclasses import dataclass
from enum import Enum
from typing import Union
import numpy as np


@dataclass
class OrbitalElements:
    orbit_type: Union[str, list[str]] = None

    # Classic Orbital elements
    major_axis: Union[float, np.typing.NDArray] = None
    eccentricity: Union[float, np.typing.NDArray] = None
    inclination: Union[float, np.typing.NDArray] = None
    ascending_node: Union[float, np.typing.NDArray] = None  # Right ascension of the ascending node
    argument_of_perigee: Union[float, np.typing.NDArray] = None
    true_anomaly: Union[float, np.typing.NDArray] = None

    # Alternative Orbital elements
    argument_of_latitude: Union[float, np.typing.NDArray] = None
    longitude_of_periapsis: Union[float, np.typing.NDArray] = None
    true_longitude: Union[float, np.typing.NDArray] = None



