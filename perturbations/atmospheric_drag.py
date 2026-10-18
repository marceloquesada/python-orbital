import numpy as np
from numba import njit
from orbital_utils import constants

_BASE_H = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
                   dtype=np.float64)                                   # km
_DENSITY = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5,
                     1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9,
                     2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12,
                     3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14,
                     5.245e-15, 3.019e-15])                            # kg/m^3
_SCALE_H = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877,
                     7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628,
                     53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00])  # km


@njit
def simple_exponential_density_model(h):
    n = _BASE_H.shape[0]
    if h <= _BASE_H[0]:
        i = 0
    elif h >= _BASE_H[n - 1]:
        i = n - 1
    else:
        i = np.searchsorted(_BASE_H, h, side='right') - 1
    return _DENSITY[i] * np.exp(-(h - _BASE_H[i]) / _SCALE_H[i])


class Atmospheric_drag_perturbation:
    def __init__(self, contact_surface_area, drag_coef, mass, model='exponential', optimization_steps = 0):
        self.surface_area = contact_surface_area
        self.drag_coef = drag_coef
        self.mass = mass
        self.last_acc = np.zeros(3)
        self.optim_steps = optimization_steps
        self.steps_counter = 0

    def get_acceleration(self, state_vector, dt=None):
        if self.steps_counter >= self.optim_steps:
            self.steps_counter = 0
            x, y, z = state_vector[0], state_vector[1], state_vector[2]
            w = constants.w_earth
        
            altitude = np.sqrt(x*x + y*y + z*z) - constants.radius_earth
        
            # v_rel = v - omega x r, with omega = [0, 0, w]
            v_rel = np.array([state_vector[3] + w*y,
                                state_vector[4] - w*x,
                                state_vector[5]])
            v_rel_norm = np.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)
        
            rho = simple_exponential_density_model(altitude)

            self.last_acc = -0.5 * self.drag_coef * (self.surface_area / self.mass) * rho * 1e3 * v_rel_norm * v_rel
        
            # a [m/s^2] = -0.5 Cd (A/m) rho v^2 v_hat  ->  km/s^mass2 gives the 1e3 factor
            return self.last_acc
        else:
            self.steps_counter += 1
            return self.last_acc