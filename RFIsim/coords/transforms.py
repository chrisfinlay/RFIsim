from skyfield.api import load
import numpy as np

def radec_to_lmn(ra, dec, phase_centre):
    """
    Convert right-ascension and declination to direction cosines.

    Parameters
    ----------
    ra : array_like (float)
        Right-ascension in degrees.
    dec : array_like (float)
        Declination in degrees.
    phase_center : array_like (float)
        The coordinates of the phase centre in degrees.

    Returns
    -------
    (l,m,n) : tuple
        l, m and n coordinates. The direction cosines.
    """
    ra, dec = np.deg2rad([ra, dec])
    phase_centre = np.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = np.cos(dec)*np.sin(delta_ra)
    m = np.sin(dec)*np.cos(dec_0) - np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)
    n = np.sqrt(1 - l**2 - m**2) - 1

    return l, m, n


def ENU_to_GPS_el(gps_centre, enu):
    """
    Convert a set of points in ENU co-ordinates to gps coordinates.

    Parameters:
    ----------
    gps_centre: array-like (3,)
        The latitude, longitude and elevation, (lat,lon), of the reference
        antenna sitting at ENU = (0,0,0).
    enu: array-like (n_ants, 3)
        The ENU co-ordinates of each antenna. East, North, Up.

    Returns:
    --------
    gps_ants: array-like (n_ants, 3)
        The GPS and elevation coordinates of each antenna.
    """

    earth_rad = 6.371e6 # metres
    d_lon = np.rad2deg(enu[:,1]/(earth_rad*np.cos(np.deg2rad(gps_centre[0]))))
    d_lat = np.rad2deg(enu[:,0]/earth_rad)
    gps_ants = np.array(gps_centre)[None,:] + \
               np.array([d_lat, d_lon, enu[:,-1]]).T

    return gps_ants


def ENU_to_ITRF(enu, lat, lon):
    """
    Calculate ITRF coordinates from ENU coordinates,
    latitude and longitude.

    Paramters:
    ----------
    enu: np.array (n_ant, 3)
        The East, North, Up coordinates of each antenna.
    lat: float
        The latitude of the observer.
    lon: float
        The longitude of the observer.

    Returns:
    --------
    itrf: np.array (n_ant, 3)
        The ITRF coordinates of the antennas.
    """

    E, L = np.deg2rad([lat, lon])
    sL, cL = np.sin(L), np.cos(L)
    sE, cE = np.sin(E), np.cos(E)

    R = np.array([[-sL, -cL*sE, cL*cE],
                  [cL,  -sL*sE, sL*cE],
                  [0.0,    cE,   sE  ]])

    itrf = np.dot(R, enu.T).T

    return itrf


def ITRF_to_UVW(ITRF, ra, dec, lon, datetime):
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    longitude a Greenwich Mean Sidereal Time.

    Parameters:
    -----------
    ITRF: np.array (n_ant, 3)
        Antenna positions in the ITRF frame in units of metres.
    ra: float
        The right ascension of the target in decimal degrees.
    dec: float
        The declination of the target in decimal degrees.
    lon: float
        The longitude at the observation location in decimal degrees.
    datetime: datetime object
        Timezone aware datetime object at which to compute uvw coordinates.

    Returns:
    --------
    uvw: np.array (n_ant, 3)
        The uvw coordinates of the antennas for a given observer
        location, time and target (ra,dec).
    """
    ts = load.timescale()
    t = ts.utc(datetime)
    gmst = t.gmst

    H0 = gmst + lon - ra
    d0 = dec

    H0, d0 = np.deg2rad([H0, d0])
    sH0, cH0 = np.sin(H0), np.cos(H0)
    sd0, cd0 = np.sin(d0), np.cos(d0)

    R = np.array([[sH0,      cH0,      0.0],
                  [-sd0*cH0, sd0*sH0,  cd0],
                  [cd0*cH0,  -cd0*sH0, sd0]])

    uvw = np.dot(R, ITRF.T).T

    return uvw


def ENU_to_UVW(enu, lat, lon, ra, dec, datetime):

    itrf = ENU_to_ITRF(enu, lat, lon)
    if isinstance(datetime, list):
        uvw = np.array([ITRF_to_UVW(itrf, ra, dec, lon, dt)
                        for dt in datetime])
    else:
        uvw = ITRF_to_UVW(itrf, ra, dec, lon, datetime)

    return uvw
