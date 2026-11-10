import datetime
import numpy as np
from numba import njit
from orbital_utils import constants, geodetic

_BASE_H = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
_DENSITY = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5,
                     1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9,
                     2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12,
                     3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14,
                     5.245e-15, 3.019e-15])
_SCALE_H = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877,
                     7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628,
                     53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00])


def simple_exponential_density_model(h):
    n = _BASE_H.shape[0]
    if h < 1:
        return 0.0

    if h <= _BASE_H[0]:
        i = 0
    elif h >= _BASE_H[n - 1]:
        i = n - 1
    else:
        i = np.searchsorted(_BASE_H, h, side='right') - 1
    return _DENSITY[i] * np.exp(-(h - _BASE_H[i]) / _SCALE_H[i]) * 10**9  # kg/km^3


class Atmospheric_drag_perturbation_exp:
    def __init__(self, drag_coef, gamma):
        self.gamma = gamma/(10**6)
        self.drag_coef = drag_coef

    def get_acceleration(self, state_vector, dt=None):
        r, v = state_vector[0:3], state_vector[3:6]
        w = constants.w_earth
    
        altitude = np.linalg.norm(r) - constants.radius_earth
    
        rho = simple_exponential_density_model(altitude)

        v_rel = v - np.cross(np.array([0.0, 0.0, constants.w_earth]), r)

        drag = -v_rel*0.5*rho*np.linalg.norm(v_rel)*self.drag_coef*self.gamma

        return drag


def td_88(h, lat, t, f_x, f_0, k_p, d):
    """
    TD-88 density (Sehnal, 1988) in kg/m^3.

    h:   altitude [km]
    lat: latitude [rad]
    t:   local solar time [h]
    f_x: daily solar flux [cm/day], f_0: mean solar flux over 3 solar rotations [cm/day]
    k_p: geomagnetic index (0..10), taken 3 h earlier
    d:   day of the year
    """
    a = np.array([0.007, 0.2875, 0.04762, 0.0471, 7.0, 7.0, 0.3333, 15.0])  # a_1..a_8
    p = np.array([263, -263, -29.41, 8.0913, 10.0813])                     # p_3..p_7
    k = np.array([
        [ 2.96815e-15,  7.66373e-9,    1.65738e-10,   3.87086e-11 ],
        [ 2.81456e-15, -4.40149e-9,    3.34283e-10,   9.35229e-11 ],
        [-1.233e-14,    1.18107e-10,  -1.47817e-10,  -1.51755e-12 ],
        [-1.14892e-17, -1.59664e-11,  -6.46708e-12,  -2.04955e-12 ],
        [-3.90064e-16, -2.407553e-10, -1.398567e-11, -3.059493e-12],
        [ 7.42439e-15,  6.43785e-11,   1.36185e-10,   3.517e-11   ],
        [-3.41594e-16,  7.44666e-12,   4.5416e-12,    2.07975e-12 ],
    ])

    f_n = (f_0 - 60)/160                  # (80)
    fx_factor = 1 + a[0]*(f_x - f_0)      # (77)
    f0_factor = a[1] + f_n                # (78)
    k_0 = 1 + a[2]*(k_p - 3)              # (79)

    # Annual (days) and diurnal (hours) phases converted to angles
    W_d = 2*np.pi/365
    W_t = 2*np.pi/24

    g_1 = 1                                                                      # (82)
    g_2 = f_n/(2 + a[3])                                                         # (83)
    g_3 = np.sin(W_d*(d - p[0]))*np.sin(lat)                                     # (84)
    g_4 = (a[4]*f_n + 1)*np.sin(W_d*(d - p[1]))                                  # (85)
    g_5 = (a[5]*f_n + 1)*np.sin(2*W_d*(d - p[2]))                                # (86)
    g_6 = (a[6]*f_n + 1)*np.sin(W_t*(t - p[3]))*np.cos(lat)                      # (87)
    g_7 = (a[7]*f_n + 1)*np.sin(2*W_t*(t - p[4]))*(np.cos(lat)**2)               # (88)

    g = np.array([g_1, g_2, g_3, g_4, g_5, g_6, g_7])

    hn_gn = 0
    for i in range(7):
        h_n = k[i, 0]                                                            # (81)
        for j in range(1, 4):
            h_n += k[i, j]*np.exp((120 - h)/(29*j))
        hn_gn += h_n*g[i]

    rho = fx_factor*f0_factor*k_0*hn_gn   # (76), kg/m^3

    return rho


class Atmospheric_drag_perturbation_td88:
    def __init__(self, drag_coef, gamma, f_x, f_0, k_p, epoch):
        """f_x, f_0: daily and mean solar flux [cm/day]; epoch: start of the propagation as Unix time (UTC seconds)."""
        self.gamma = gamma/(10**6)
        self.drag_coef = drag_coef
        self.epoch = epoch
        self.f_x = f_x
        self.f_0 = f_0
        self.k_p = k_p

    def get_acceleration(self, state_vector, dt):
        # dt is the elapsed time since the start of the propagation (not an increment)
        utc = datetime.datetime.fromtimestamp(self.epoch + dt, datetime.timezone.utc)
        utc_hours = utc.hour + utc.minute/60 + (utc.second + utc.microsecond*1e-6)/3600
        d = utc.timetuple().tm_yday + utc_hours/24

        lat, lon, alt = geodetic.state_vector_to_lat_lon(state_vector, dt, self.epoch)
        local_time = np.mod(utc_hours + lon/15, 24)

        r, v = state_vector[0:3], state_vector[3:6]

        altitude = np.linalg.norm(r) - constants.radius_earth

        rho = td_88(altitude, np.deg2rad(lat), local_time, self.f_x, self.f_0, self.k_p, d)*10**9  # kg/km^3

        v_rel = v - np.cross(np.array([0.0, 0.0, constants.w_earth]), r)

        drag = -v_rel*0.5*rho*np.linalg.norm(v_rel)*self.drag_coef*self.gamma

        return drag