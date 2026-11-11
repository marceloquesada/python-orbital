import datetime
import numpy as np
import pymsis
from numba import njit
from orbital_utils import constants, geodetic

# US Standard Atmosphere 1976, exponential interpolation (Curtis, Orbital Mechanics for
# Engineering Students, Appendix D.39 "atmosphere.m")
_BASE_H = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])  # km
_DENSITY = np.array([1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
                     1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
                     2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
                     1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])  # kg/m^3
_SCALE_H = np.array([7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
                     5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
                     21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
                     60.980, 65.654, 76.377, 100.587, 147.203, 208.020])  # km, one per interval (27)


def simple_exponential_density_model(h):
    """Density [kg/km^3] at geometric altitude h [km], USSA76 exponential interpolation (Curtis)."""
    # Altitudes outside 0-1000 km are clamped, as in Curtis
    z = min(max(h, _BASE_H[0]), _BASE_H[-1])

    # Interval i such that _BASE_H[i] <= z < _BASE_H[i+1]; z = 1000 km uses the last interval
    i = min(np.searchsorted(_BASE_H, z, side='right') - 1, _SCALE_H.shape[0] - 1)

    return _DENSITY[i] * np.exp(-(z - _BASE_H[i]) / _SCALE_H[i]) * 10**9  # kg/m^3 -> kg/km^3


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


def ips_density_model(h, f107, ap):
    """
    Density [kg/km^3] at altitude h [km] from the IPS Radio and Space Services (Australian Space
    Weather Agency) model, "Satellite Orbital Decay Calculations" (Kennewell, updated by R. Panwar):

        T = 900 + 2.5 (F10.7 - 70) + 1.5 Ap        [K]
        m = 27 - 0.012 (h - 200)
        H = T / m                                   [km]
        rho = 6e-10 exp(-(h - 175) / H)             [kg/m^3]

    Fitted for 180 < h < 500 km (de Vries, LLNL-TR-453771, also uses it up to 700 km).
    f107: solar radio flux, usually the 90-day average [sfu]; ap: geomagnetic Ap index.
    """
    T = 900 + 2.5*(f107 - 70) + 1.5*ap
    m = 27 - 0.012*(h - 200)
    if m <= 0:  # Formula breaks down (H -> infinity) above ~2450 km; no drag that high
        return 0.0
    H = T/m

    return 6e-10*np.exp(-(h - 175)/H)*10**9  # kg/m^3 -> kg/km^3


_IPS_ARTICLE_RE = 6378000.0  # Earth radius used by the IPS article program [m]


def ips_article_altitude(h):
    """
    Altitude [km] that the IPS article's QBASIC program "sees" for a true altitude h [km].

    The program recovers the orbital radius from the period with R = (G Me P^2 / 4 pi^2) ^ .33333
    instead of ^(1/3), i.e. R_used = R^0.99999 (R in metres), which places the satellite ~1.05 km
    lower than it is. Since the density is evaluated at this lower altitude, the published decay
    table and the "Re-entry after 46 days" result decay faster than the model itself predicts.
    """
    return ((_IPS_ARTICLE_RE + h*1e3)**(3*0.33333) - _IPS_ARTICLE_RE)/1e3


class Atmospheric_drag_perturbation_ips:
    def __init__(self, drag_coef, gamma, f107, ap, rotating_atmosphere=True, emulate_article_rounding=False):
        """
        Drag with the IPS exponential, variable-scale-height density (see ips_density_model).

        drag_coef:                C_D
        gamma:                    area-to-mass ratio [m^2/kg]
        f107:                     solar radio flux (90-day average) [sfu]
        ap:                       geomagnetic Ap index
        rotating_atmosphere:      if False, drag uses the inertial velocity (as in the IPS worked example)
        emulate_article_rounding: if True, evaluate the density at ips_article_altitude(h) to reproduce the
                                  cube-root rounding error of the IPS article program (and its published table)
        """
        self.gamma = gamma/(10**6)  # km^2/kg
        self.drag_coef = drag_coef
        self.f107 = f107
        self.ap = ap
        self.rotating_atmosphere = rotating_atmosphere
        self.emulate_article_rounding = emulate_article_rounding

    def get_acceleration(self, state_vector, dt=None):
        r, v = state_vector[0:3], state_vector[3:6]

        altitude = np.linalg.norm(r) - constants.radius_earth
        if self.emulate_article_rounding:
            altitude = ips_article_altitude(altitude)

        rho = ips_density_model(altitude, self.f107, self.ap)

        if self.rotating_atmosphere:
            v_rel = v - np.cross(np.array([0.0, 0.0, constants.w_earth]), r)
        else:
            v_rel = v

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

# Kp (0..9, integer thirds allowed) -> ap [2 nT] conversion table (Bartels)
_KP_TABLE = np.array([0, 1/3, 2/3, 1, 4/3, 5/3, 2, 7/3, 8/3, 3, 10/3, 11/3, 4, 13/3, 14/3,
                      5, 16/3, 17/3, 6, 19/3, 20/3, 7, 22/3, 23/3, 8, 25/3, 26/3, 9])
_AP_TABLE = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39,
                      48, 56, 67, 80, 94, 111, 132, 154, 179, 207, 236, 300, 400])


def kp_to_ap(k_p):
    """Convert a planetary Kp index to the equivalent ap index (linear interpolation of the Bartels table)."""
    return float(np.interp(k_p, _KP_TABLE, _AP_TABLE))


class Atmospheric_drag_perturbation_msis:
    def __init__(self, drag_coef, gamma, f107, f107a, ap, epoch, version=2.1):
        """
        Drag with NRLMSIS density (pymsis).

        drag_coef: C_D
        gamma:     area-to-mass ratio [m^2/kg]
        f107:      daily F10.7 solar flux (previous day) [sfu]
        f107a:     81-day centred average F10.7 [sfu]
        ap:        daily geomagnetic ap index (use kp_to_ap() to convert from Kp)
        epoch:     start of the propagation as Unix time (UTC seconds)
        version:   MSIS version passed to pymsis (2.1, 2.0, or 0 for NRLMSISE-00)
        """
        self.gamma = gamma/(10**6)  # km^2/kg
        self.drag_coef = drag_coef
        self.epoch = epoch
        self.f107 = f107
        self.f107a = f107a
        self.aps = [[ap]*7]
        self.version = version

    def get_acceleration(self, state_vector, dt):
        # dt is the elapsed time since the start of the propagation (not an increment)
        lat, lon, alt = geodetic.state_vector_to_lat_lon(state_vector, dt, self.epoch)
        date = np.datetime64(int(round((self.epoch + dt)*1e6)), 'us')

        rho = pymsis.calculate(date, lon, lat, alt, f107s=self.f107, f107as=self.f107a,
                               aps=self.aps, version=self.version)
        rho = float(np.squeeze(rho)[pymsis.Variable.MASS_DENSITY])*10**9  # kg/m^3 -> kg/km^3

        r, v = state_vector[0:3], state_vector[3:6]
        v_rel = v - np.cross(np.array([0.0, 0.0, constants.w_earth]), r)

        drag = -v_rel*0.5*rho*np.linalg.norm(v_rel)*self.drag_coef*self.gamma

        return drag
