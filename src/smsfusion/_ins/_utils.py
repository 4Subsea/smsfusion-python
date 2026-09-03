import numpy as np

from ._common import _signed_smallest_angle


def euler_from_acc(f, nav_frame, yaw=0.0, degrees=False):
    """
    Estimate roll and pitch angles from specific force (i.e., accelerometer) measurement.
    As yaw cannot be determined from specific force alone, the keyword argument ``yaw``
    is returned.

    Parameters
    ----------
    f: array-like
        Specific force (i.e., acceleration) measurement vector (fx, fy, fz).
    nav_frame: {'NED', 'ENU'}
        Navigation frame. Should be either 'NED' or 'ENU'.
    yaw: float, optional
        Yaw value in radians to be returned. Default is 0.0.
    degrees : bool, optional
        Whether to return the Euler angles in degrees or radians. Default is ``False``
        (radians).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Euler angles (roll, pitch, yaw).
    """
    fx, fy, fz = f

    if nav_frame.lower() == "ned":
        roll = np.arctan2(-fy, -fz)
        pitch = np.arctan2(fx, np.sqrt(fy**2 + fz**2))
    elif nav_frame.lower() == "enu":
        roll = np.arctan2(fy, fz)
        pitch = -np.arctan2(fx, np.sqrt(fy**2 + fz**2))
    else:
        raise ValueError("Invalid navigation frame. Should be 'NED' or 'ENU'.")

    theta = np.array([roll, pitch, yaw])

    if degrees:
        theta = (180.0 / np.pi) * theta

    return theta


def gravity(lat: float | None = None, degrees: bool = True) -> float:
    """
    Calculates the gravitational acceleration based on the World Geodetic System
    (1984) Ellipsoidal Gravity Formula (WGS-84).

    The WGS-84 formula is given by::

        g = g_e * (1 - k * sin(lat)^2) / sqrt(1 - e^2 * sin(lat)^2)

    where, ::

        g_e = 9.780325335903891718546
        k = 0.00193185265245827352087
        e^2 = 0.006694379990141316996137

    and ``lat`` is the latitude.

    If no latitude is provided, the 'standard gravity', ``g_0``, is returned instead.
    The standard gravity is by definition of the ISO/IEC 8000 given as
    ``g_0 = 9.80665``.

    Parameters
    ----------
    lat : float, optional
        Latitude. If none provided, the 'standard gravity' is returned.
    degrees : bool, optional
        Specify whether the latitude, ``lat``, is in degrees or radians.
        Applicapble only if ``lat`` is provided.
    """
    if lat is None:
        g_0 = 9.80665  # standard gravity in m/s^2
        return g_0

    g_e = 9.780325335903891718546  # gravity at equator
    k = 0.00193185265245827352087  # formula constant
    e_2 = 0.006694379990141316996137  # spheroid's squared eccentricity

    if degrees:
        lat = (np.pi / 180.0) * lat

    g = g_e * (1.0 + k * np.sin(lat) ** 2.0) / np.sqrt(1.0 - e_2 * np.sin(lat) ** 2.0)
    return g  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


class FixedNED:
    """
    Convert position coordinates between a fixed NED frame (x, y, z) and ECEF frame
    (lattitude, longitude, height).

    The fixed NED frame is a tangential plane on the WGS-84 ellipsoid with its origin
    fixed at the provided reference coordinates. It is assumed that the tangential
    plane is close to the ellipsoid surface.

    Parameters
    ----------
    lat_ref: float
        Reference latitude coordinate in decimal degrees.
    lon_ref: float
        Reference longitude coordinate in decimal degrees.
    height_ref: ref
        Reference height coordinate in decimal degrees.
    """

    def __init__(self, lat_ref: float, lon_ref: float, height_ref: float) -> None:
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref
        self._height_ref = height_ref

        radius_eq = 6_378_137  # equatorial radius (WGS-84)
        radius_polar = 6_356_752.314245  # polar radius (WGS-84)

        radius_ratio_squared = (radius_polar / radius_eq) ** 2
        denom = np.cos(
            self._lat_ref * (np.pi / 180.0)
        ) ** 2 + radius_ratio_squared * np.sin(self._lat_ref * (np.pi / 180.0))

        self._Rn = radius_eq / np.sqrt(denom)  # radius prime vertical
        self._Rm = self._Rn * radius_ratio_squared / denom  # radius meridian

        self._Rm_h = self._Rm + height_ref
        self._Rn_h_cos = (self._Rn + height_ref) * np.cos(
            self._lat_ref * (np.pi / 180.0)
        )

    def to_llh(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Compute longitude, latitude, and height coordinates (WGS-84) from local
        Cartesian coordinates in the fixed NED frame.

        Parameters
        ----------
        x: float
            Local x-coordinate in meters in the fixed NED frame.
        y: float
            Local y-coordinate in meters in the fixed NED frame.
        z: float
            Local z-coordinate in meters in the fixed NED frame.

        Returns
        -------
        lat: float
            Latitude coordinate in decimal degrees.
        lon: float
            Longitude coordinate in decimal degrees.
        height: float
            Height coordinate in meters.
        """
        dlat = (180.0 / np.pi) * x / self._Rm_h
        dlon = (180.0 / np.pi) * y / self._Rn_h_cos

        lat = _signed_smallest_angle(self._lat_ref + dlat, degrees=True)
        lon = _signed_smallest_angle(self._lon_ref + dlon, degrees=True)
        h = self._height_ref - z
        return lat, lon, h

    def to_xyz(
        self, lat: float, lon: float, height: float
    ) -> tuple[float, float, float]:
        """
        Compute local Cartesian coordinates in the fixed NED frame from longitude,
        latitude, and height coordinates (WGS-84).

        Parameters
        ----------
        lat: float
            Latitude coordinate in decimal degrees.
        lon: float
            Longitude coordinate in decimal degrees.
        height: float
            Height coordinate in meters.

        Returns
        -------
        x: float
            Local x-coordinate in meters in the fixed NED frame.
        y: float
            Local y-coordinate in meters in the fixed NED frame.
        z: float
            Local z-coordinate in meters in the fixed NED frame.
        """
        dlat = lat - self._lat_ref
        dlon = lon - self._lon_ref

        x = (np.pi / 180.0) * dlat * self._Rm_h
        y = (np.pi / 180.0) * dlon * self._Rn_h_cos
        z = self._height_ref - height
        return x, y, z
