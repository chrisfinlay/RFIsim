import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from skyfield.api import load, Topos
from skyfield.positionlib import ICRF, Angle
import pandas as pd
import numpy as np
import datetime
import os

def get_archival_tles(start_date, end_date):
    """
    Collect TLEs for a specific date range.

    Parameters
    ----------
    start_date: Datetime object
        Start date and time of observation.
    end_date: Datetime object
        End date and time of observation.

    Returns
    -------
    file_path: str
        File path to TLE file.
    """

    tle_file = start_date.strftime('%Y-%m-%d')+'.tle'
    file_path = os.path.join('TLEs', tle_file)

    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            print('File {} exists and is readable.'.format(tle_file))
            return file_path

    st = SpaceTrackClient('cfinlay@ska.ac.za', 'SARAORFIsimulator')

    drange = op.inclusive_range(start_date, end_date)

    tles = [x.encode() for x in st.tle(iter_lines=True, epoch=drange,
                                       orderby='TLE_LINE1', format='tle')]

    ids = []
    for i in range(len(tles)/2):
        ids.append(int(tles[1::2][i].split()[1]))

    tles = pd.DataFrame(np.array([[tles[::2], tles[1::2]]])[0].T, index=ids)

    tles = tles[~tles.index.duplicated()]
    tles.index.name = 'NORAD_ID'
    tles.columns = ['E1', 'E2']

    sat_names = pd.read_csv('TLEs/sat_names.csv', index_col='NORAD_ID')
    new_index = []
    for i in range(len(sat_names)):
        new_index.append(int(sat_names.index.values[i].split()[-1]))
    sat_names.index = new_index

    merged = pd.concat([tles, sat_names], axis=1, join='inner')

    print('\nWriting TLE file to {}\n'.format(file_path))

    with open(file_path, 'w') as f:
        for i in range(len(merged)):
            f.write('\n'.join(list(merged.iloc[i].values[[-1,0,1]]))+'\n')

    return file_path

def get_dist_and_sep(sat_tle, ant_gps, pointing, time):
    """
    Get the distance from an antenna to a satellite and the angular
    separation between the satellite and the antenna pointing.

    Parameters:
    -----------
    sat_tle: Skyfield.EarthSatellite
        The TLE object of a specific satellite.
    ant_gps: array-like (3,)
        The latitude, longitude and elevation of a given antenna.
    pointing: float
        The altitude of the pointing direction.
    time: datetime.datetime
        Date and time at which to evaluate.

    Returns:
    --------
    distance: float
        The distance, in meters, from the antenna to the satellite.
    ang_sep: float
        The angular separation, in degrees, between the pointing direction and the satellite.
    """

    date = time.strftime('%Y-%m-%d %H:%M:%S').split(' ')
    date = [int(x) for x in date[0].split('-')] + [int(x) for x in date[1].split(':')]

    ts = load.timescale()
    t = ts.utc(date[0], date[1], date[2], date[3]+2, date[4], date[5])

    obs = Topos(latitude_degrees=ant_gps[0], longitude_degrees=ant_gps[1],
                elevation_m=ant_gps[2])
    zenith = Topos(latitude_degrees=ant_gps[0]+pointing,
                   longitude_degrees=ant_gps[1], elevation_m=1.e30).at(t)

    topocentric = (sat_tle-obs).at(t)

    el, distance = topocentric.altaz()[0::2]
    ang_sep = np.rad2deg(topocentric.separation_from(zenith).radians)

    return np.rad2deg(el.radians), distance.m, ang_sep

def enu_to_gps_el(gps_centre, enu):
    """
    Convert a set of points in ENU co-ordinates to gps coordinates.

    Parameters:
    ----------
    gps_centre: array-like (2,)
        The latitude and longitude, (lat,lon), of the reference antenna sitting at ENU = (0,0,-).
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
    gps_ants = np.array(gps_centre)[None,:] + np.array([d_lat, d_lon]).T

    return np.concatenate([gps_ants, enu[:,-1:]], axis=1)

def get_time_delays(distances):
    """
    Get the time delay for each baseline formed by each antenna pair.

    Parameters:
    -----------
    distances: array-like (n_ants,)
        The distance between an RFI source and each antenna in metres.

    Returns:
    --------
    delays: array-like (n_bl,)
        The time delay of the time of arrival of the RFI signal between
        each pair of antennas.
    """

    c = 2.99792458e8
    a1, a2 = np.triu_indices(len(distances), 1)
    delays = (distances[a1]-distances[a2])/c

    return delays

def sinc_beam(ang_sep, params):
    """
    A sinc beam that narrows with increasing frequency.
    A parameter $\alpha \propto D$ is present to control
    the width of the beam.

    Paramters:
    ----------
    ang_sep: np.array (n_srcs,)
        The radius at which to evaluate the beam function.
        Angular separation, in degrees, from the pointing direction.
    params: array-like (2,)
        Contains alpha, the width parameter, proportional to the
        dish diameter. And the frequency. (alpha, freq)

    Returns:
    --------
    beam: np.array (n_srcs,n_freqs)
        The attenuation due to the beam at the given angular separation.
    """

    alpha, freq = params
    beam = np.sinc(alpha*freq[None,:]*ang_sep[:,None])

    return beam

def pol_beam(auto_beam, cross_beam, params, ang_sep):
    """
    Calculate the attenuation due to the primary beam for a
    given angular separation from the pointing direction.

    Parameters:
    -----------
    auto_beam: func
        Function that computes the auto-polarization beam attenuation.
        auto_beam(ang_sep, params)
    cross_beam: func
        Function that computes the auto-polarization beam attenuation.
        cross_beam(ang_sep, params)
    params: array-like
        The parameters to be used in the evaluation of the beams.
    ang_sep: np.array (n_srcs,)
        The angular separation between the source and the pointing
        direction in degrees.

    Returns:
    --------
    pol_beam: np.array (2,2,n_srcs,n_freqs)
        The polarized beam attenuation. [[HH, HV], [VH, VV]].
    """

    HH = auto_beam(ang_sep, params)
    HV = np.zeros(HH.shape)
    VH = np.zeros(HH.shape)
    VV = auto_beam(ang_sep, params)

#     Currently assuming HH = VV and both HV and VH are 0
    pol_beam = np.array([[HH, HV],[VH, VV]])

    return pol_beam

def brightness_matrix(I, Q, U, V):
    """
    Generate a brightness matrix from Stokes parameters.

    Parameters:
    -----------
    I: np.array (n_srcs,n_freqs)
        Stokes I.
    Q: np.array (n_srcs,n_freqs)
        Stokes Q.
    U: np.array (n_srcs,n_freqs)
        Stokes U.
    V: np.array (n_srcs,n_freqs)
        Stokes V.

    Returns:
    --------
    B: np.array (2,2,n_srcs,n_freqs)
        The brightness matrix.
    """

    HH = I + Q
    HV = U + 1.j*V
    VH = U - 1.j*V
    VV = I - Q

    B = np.array([[HH, HV], [VH, VV]])

    return B

def RIME(B, E, delays, freqs):
    """
    Calculate the visibilities for each baseline and frequency.

    Parameters:
    -----------
    B: np.array (2,2,n_srcs,n_freqs)
        The brightness matrix for all sources at each frequency.
    E: np.array (2,2,n_srcs,n_freqs)
        The polarized beam term for all sources at each frequency.
    delays: np.array (n_bl,n_srcs)
        The time delays, in seconds, on each baseline for all sources.
    freqs: np.array (n_freqs)
        The frequencies, in Hertz, at which we are observing.

    Returns:
    --------
    V: np.array (2,2,n_bl,n_freqs)
        The visibilities according to the RIME formalism.
    """

#     Phase delays (n_bl, n_srcs, n_freqs)
    KK = np.exp(-2.j*np.pi*freqs[None,None,:]*delays[:,:,None])

#     Source coherencies (2, 2, n_bl, n_srcs, n_freqs)
    X = B[:,:,None,:,:] * KK[None,None,:,:,:]

#     Visibilites (2, 2, n_bl, n_freqs)
    V = np.sum(np.sum(np.sum(E[:,:,None,None,:,:]*X[None,:,:,:,:], axis=1)[:,:,None,:,:] *
               np.transpose(E.conjugate(), (1,0,2,3))[None,:,:,None,:,:], axis=1), axis=3)

    return V
