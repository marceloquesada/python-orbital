"""
Latitude / longitude from inertial state vectors.

Pipeline:
    ECI (inertial, km)  --R_z(GMST)-->  ECEF (Earth-fixed, km)  --WGS-84-->  geodetic lat/lon/alt

Notes
-----
- The inertial frame is treated as a "true-of-date"-like frame (e.g. TEME, which is what
  TLE-derived states live in). Only Earth rotation (GMST) is applied; precession, nutation
  and polar motion are neglected. For a J2000 state this introduces an error of a few
  tenths of a degree in longitude, which is fine for ground tracks.
- All times are Unix epoch seconds (UTC, seconds since 1970-01-01 00:00:00); UT1 - UTC (< 0.9 s)
  and leap seconds are ignored.
"""
import calendar
from typing import Union

import numpy as np

from orbital_utils import constants
from orbital_utils import ref_systems


# WGS-84 ellipsoid
WGS84_A = constants.radius_earth          # km, equatorial radius
WGS84_F = 1 / 298.257223563               # flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)        # first eccentricity squared

J2000_JD = 2451545.0                      # Julian date of J2000.0 epoch (2000-01-01 12:00)
UNIX_EPOCH_JD = 2440587.5                 # Julian date of the Unix epoch (1970-01-01 00:00 UTC)


# TIME
def julian_date(epoch: float) -> float:
    """Julian date of a Unix epoch time (seconds since 1970-01-01 00:00:00 UTC)."""
    return UNIX_EPOCH_JD + epoch / 86400.0


def gmst(jd: Union[float, np.typing.NDArray]) -> Union[float, np.typing.NDArray]:
    """Greenwich Mean Sidereal Time in degrees [0, 360) (IAU 1982 model)."""
    T = (np.asarray(jd) - J2000_JD) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3)
    return np.mod(gmst_sec / 240.0, 360.0)  # 240 s of time = 1 deg


def tle_epoch_to_unix(TLE: list[str]) -> float:
    """Epoch of a TLE (line 1, columns 19-32) as Unix epoch seconds (UTC)."""
    epoch_field = TLE[0][18:32]
    year = int(epoch_field[:2])
    year += 2000 if year < 57 else 1900
    day_of_year = float(epoch_field[2:])
    year_start = calendar.timegm((year, 1, 1, 0, 0, 0))
    return year_start + (day_of_year - 1) * 86400.0


# FRAMES
def eci_to_ecef(r_eci: np.typing.NDArray, jd: Union[float, np.typing.NDArray]) -> np.typing.NDArray:
    """
    Rotate inertial positions to the Earth-fixed frame.

    r_eci: shape (3,) or (3, N)
    jd:    scalar, or shape (N,) matching the columns of r_eci
    """
    r_eci = np.asarray(r_eci, dtype=float)
    theta = gmst(jd)

    if np.ndim(theta) == 0:
        # ECEF = R_z(-GMST) @ ECI  (rotating the frame, not the vector)
        return ref_systems.rot_Z(float(theta), clockwise=True) @ r_eci

    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    x, y, z = r_eci
    return np.array([c * x + s * y, -s * x + c * y, z])


def ecef_to_geodetic(r_ecef: np.typing.NDArray, tol: float = 1e-12, max_iter: int = 20):
    """
    Convert Earth-fixed positions (km) to geodetic latitude, longitude (deg) and altitude (km)
    on the WGS-84 ellipsoid, using the iterative Bowring/Newton scheme.

    r_ecef: shape (3,) or (3, N)
    Returns (lat_deg, lon_deg, alt_km), each scalar or shape (N,).
    """
    x, y, z = np.asarray(r_ecef, dtype=float)
    p = np.hypot(x, y)

    lon = np.arctan2(y, x)

    # Initial guess: spherical latitude corrected for ellipsoid
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(max_iter):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        converged = np.all(np.abs(lat_new - lat) < tol)
        lat = lat_new
        if converged:
            break

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    cos_lat = np.cos(lat)
    # Near the poles p/cos(lat) is ill-conditioned; use the z-based formula there
    alt = np.where(np.abs(cos_lat) > 1e-10,
                   p / np.where(np.abs(cos_lat) > 1e-10, cos_lat, 1.0) - N,
                   np.abs(z) - N * (1 - WGS84_E2))

    lat_deg, lon_deg = np.rad2deg(lat), np.rad2deg(lon)
    if np.ndim(lat_deg) == 0:
        return float(lat_deg), float(lon_deg), float(alt)
    return lat_deg, lon_deg, alt


# MAIN ENTRY POINT
def state_vector_to_lat_lon(state_vector: np.typing.NDArray, t: float, t_0: float):
    """
    Geodetic latitude/longitude/altitude of a satellite from a single inertial state vector.

    state_vector: 6x1 state [x, y, z, vx, vy, vz] (km, km/s); shape (6,) or (6, 1).
                  Only the position is used.
    t:            current time, in seconds since t_0 (same time axis as the propagators).
    t_0:          initial time as Unix epoch seconds (UTC), e.g. 1759403557.52.

    Returns (lat_deg, lon_deg, alt_km). Longitude is in [-180, 180].
    """
    X = np.asarray(state_vector, dtype=float)
    if X.size != 6 or X.shape not in ((6,), (6, 1)):
        raise ValueError(f"state_vector must be 6x1, got shape {X.shape}.")
    r_eci = X.reshape(6)[0:3]

    jd = julian_date(t_0) + float(t) / 86400.0

    r_ecef = eci_to_ecef(r_eci, jd)
    return ecef_to_geodetic(r_ecef)


if __name__ == "__main__":
    from orbital_elements import tle

    iss_tle = ["1 25544U 98067A   25275.46710094  .00015763  00000+0  28741-3 0  9998",
               "2 25544  51.6327 135.3521 0001043 189.2069 170.8900 15.49620641531809"]

    epoch = tle_epoch_to_unix(iss_tle)
    X0 = tle.TLE_to_state_vectors(iss_tle, constants.mu_earth)

    lat, lon, alt = state_vector_to_lat_lon(X0, 0.0, epoch)
    print(f"Epoch (Unix, UTC): {epoch:.3f} s")
    print(f"Latitude:  {lat:9.4f} deg")
    print(f"Longitude: {lon:9.4f} deg")
    print(f"Altitude:  {alt:9.3f} km")
