import numpy as np
from orbital_utils import constants


def simple_exponential_density_model(h):
    density = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5,
               1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9,
               2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12,
               6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15], dtype=np.float128)
    base_h = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
              150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], dtype=np.float128)
    scale_h = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
               22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00], dtype=np.float128)

    if h < base_h[len(base_h) - 1]:
        i = np.argmax(base_h > h)
    else:
        i = len(base_h) - 1

    h_0 = base_h[i]
    rho_0 = density[i]
    H = scale_h[i]
    
    rho = rho_0*np.exp(-(h - h_0)/(H))

    return rho


def get_local_atmosphere_velocity_vector(state_vector):
    r_vec = state_vector[0:2]

    v_vec = np.array([-constants.w_earth*r_vec[1], constants.w_earth*r_vec[0], 0], dtype=np.float128)

    return v_vec


class Atmospheric_drag_perturbation():
    def __init__(self, contact_surface_area, drag_coef, mass, model='exponential'):  # Other models not yet implemented
        self.surface_area = contact_surface_area
        self.drag_coef = drag_coef
        self.mass = mass

        if model == 'exponential':
            self.get_density = simple_exponential_density_model


    def get_acceleration(self, state_vector, dt):
        radius = np.linalg.norm(state_vector[0:3])
        vel = np.linalg.norm(state_vector[3:6])

        altitude = radius - constants.radius_earth

        vel_atmosphere = get_local_atmosphere_velocity_vector(state_vector)
        vel_rel = vel + vel_atmosphere
        vel_rel_norm = np.linalg.norm(vel_rel)

        density = self.get_density(altitude)

        part_1 = density*(vel_rel_norm**2)*(vel_rel/vel_rel_norm)
        part_2 = -0.5*self.drag_coef*(self.surface_area/self.mass)

        # a_drag = -0.5*density*self.drag_coef*(self.surface_area/self.mass)*(vel_rel_norm**2)*(vel_rel/vel_rel_norm)
        a_drag = part_1*part_2

        # print(f"radius : {radius:.3f}", end=', ')
        # print(f"vel : {np.linalg.norm(vel):.3f}", end=', ')
        # print(f"vel_atm : {np.linalg.norm(vel_atmosphere):.3f}", end=', ')
        # print(f"vel_rel : {vel_rel_norm:.3f}", end=', ')
        # print(f"a_drag_scalar : {np.linalg.norm(a_drag)}", end=', ')
        # print(f"part_1 : {part_1}", end=', ')
        # print(f"part_2 : {part_2}", end=', ')
        # print(f"-0.5*rho*cd*A/m : {-0.5*density*self.drag_coef*(self.surface_area/self.mass)}", end=', ')
        # print(f"v_rel**2 * v_rel_vec/|v_rel| : {(vel_rel_norm**2)*(vel_rel/vel_rel_norm)}", end=', ')


        return a_drag
    