import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from skyfield.api import load as sf_load
from skyfield.api import Topos
# from skyfield.positionlib import ICRF, Angle
from RFIsim.coords.metrics import angular_separation
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

    tle_dir = os.path.dirname(os.path.abspath(__file__))
    tle_dir = os.path.join(tle_dir, 'TLEs')
#     tle_dir = '/home/chris/HIRAX/RFIsim/utils/rfi/sat_sim/TLEs'
    file_path = os.path.join(tle_dir, tle_file)

    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            print('File {} exists and is readable.'.format(tle_file))
            return file_path

    st = SpaceTrackClient('cfinlay@ska.ac.za', 'SARAORFIsimulator')

    drange = op.inclusive_range(start_date, end_date)

    norad_ids = list(np.load('norad_ids.npy'))

    tles = [x for x in st.tle(iter_lines=True, epoch=drange, orderby='TLE_LINE1',
                              format='tle', norad_cat_id=norad_ids)]
    # x.encode() for python 2

    ids = []
    for i in range(len(tles)//2):
        ids.append(int(tles[1::2][i].split()[1]))

    tles = pd.DataFrame(np.array([[tles[::2], tles[1::2]]])[0].T, index=ids)

    tles = tles[~tles.index.duplicated()]
    tles.index.name = 'NORAD_ID'
    tles.columns = ['E1', 'E2']

    sat_names_path = os.path.join(tle_dir, 'sat_names.csv')
    sat_names = pd.read_csv(sat_names_path, index_col='NORAD_ID')
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

def get_norad_ids(sat_class='all', save=True, remove_tles=True):
    """
    Get the NORAD IDs for a group of satellites.

    Parameters:
    -----------
    sat_class: str
        Class of satellites to get IDs for.
        Options -

    Returns:
    --------
    ids: list
        List of NORAD IDs.
    """
    sat_classes = {'gps': 'gps-ops.txt',
                   'glonass': 'glo-ops.txt',
                   'galileo': 'galileo.txt',
                   'beidou': 'beidou.txt',
                   'sbas': 'sbas.txt',
                   'nnss': 'nnss.txt',
                   'leo': 'musson.txt',
                   'geosynchronous': 'geo.txt',
#                    'geo_protected': 'gpz.php',
#                    'geo_protected_plus': 'gpz-plus.php',
                   'intelsat': 'intelsat.txt',
                   'iridium': 'iridium.txt',
                   'iridium_NEXT': 'iridium-NEXT.txt',
                   'starlink': 'starlink.txt',
                   'orbcomm': 'orbcomm.txt',
                   'ses': 'ses.txt',
                   'global_star': 'globalstar.txt',
                  }


    norad_url = 'https://www.celestrak.com/NORAD/elements/'

    if sat_class=='all':
        ids = []
        for i in range(len(sat_classes)):
            tle_url = os.path.join(norad_url, sat_classes[list(sat_classes.keys())[i]])
            ids += list(sf_load.tle(tle_url).keys())[::4]
            os.remove(sat_classes[list(sat_classes.keys())[i]])
    else:
        tle_url = os.path.join(norad_url, sat_classes[sat_class])
        ids = list(sf_load.tle(tle_url).keys())[::4]
        os.remove(sat_classes[sat_class])

    ids = np.array([x for x in ids if isinstance(x, int)])

    if save:
        np.save('norad_ids.npy', ids)

    return ids

def get_dist_and_seps(sat_tles, ant_locs, times, target):
    """
    Get the distance from an antenna to a satellite and the angular
    separation between the satellite and the antenna pointing.

    Parameters:
    -----------
    sat_tles: Skyfield.EarthSatellite
        The TLE objects of a set of satellite.
    ant_locs: array-like (3,)
        The latitude, longitude and elevation of all antennas.
    times: datetime.datetime
        Dates and times at which to evaluate.
    target: array-like
        Right ascension and declination of the target.

    Returns:
    --------
    el_dist_ra_dec_sep: np.array (n_time, n_ant, n_sats, 5)
        The distance, in meters, from the antenna to the satellite.
    """

    ants = [Topos(latitude_degrees=ant_locs[i,0], longitude_degrees=ant_locs[i,1],
                elevation_m=ant_locs[i,2]) for i in range(len(ant_locs))]

    ts = sf_load.timescale()
    t = ts.utc(times)

    el_dist_ra_dec_sep = np.zeros((len(times), len(ants), len(sat_tles), 5))

    for i, sat_tle in enumerate(sat_tles):
        for j, ant in enumerate(ants):
            topocentric = (sat_tle-ant).at(t)
            el = topocentric.altaz()[0].degrees
            dist = topocentric.altaz()[2].m
            ra = topocentric.radec()[0]._degrees
            dec = topocentric.radec()[1]._degrees
            el_dist_ra_dec_sep[:,j,i,:4] = np.array([el, dist, ra, dec]).T

    ra, dec = el_dist_ra_dec_sep[:,:,:,2], el_dist_ra_dec_sep[:,:,:,3]
    el_dist_ra_dec_sep[:,:,:,-1] = angular_separation(ra, dec, target)

    return el_dist_ra_dec_sep

# def get_dist_and_sep(sat_tle, ant_gps, pointing, time):
#     """
#     Get the distance from an antenna to a satellite and the angular
#     separation between the satellite and the antenna pointing.
#
#     Parameters:
#     -----------
#     sat_tle: Skyfield.EarthSatellite
#         The TLE object of a specific satellite.
#     ant_gps: array-like (3,)
#         The latitude, longitude and elevation of a given antenna.
#     pointing: float
#         The altitude of the pointing direction.
#     time: datetime.datetime
#         Date and time at which to evaluate.
#
#     Returns:
#     --------
#     distance: float
#         The distance, in meters, from the antenna to the satellite.
#     ang_sep: float
#         The angular separation, in degrees, between the pointing direction and the satellite.
#     """
#
#     date = time.strftime('%Y-%m-%d %H:%M:%S').split(' ')
#     date = [int(x) for x in date[0].split('-')] + [int(x) for x in date[1].split(':')]
#
#     ts = sf_load.timescale()
#     t = ts.utc(date[0], date[1], date[2], date[3]+2, date[4], date[5])
#
#     obs = Topos(latitude_degrees=ant_gps[0], longitude_degrees=ant_gps[1],
#                 elevation_m=ant_gps[2])
#     zenith = Topos(latitude_degrees=ant_gps[0]+pointing,
#                    longitude_degrees=ant_gps[1], elevation_m=1.e30).at(t)
#
#     topocentric = (sat_tle-obs).at(t)
#
#     el, distance = topocentric.altaz()[0::2]
#     ang_sep = np.rad2deg(topocentric.separation_from(zenith).radians)
#
#     return np.rad2deg(el.radians), distance.m, ang_sep

def get_time_delays(distances):
    """
    Get the time delay for each baseline formed by each antenna pair.

    Parameters:
    -----------
    distances: array-like (n_time,n_src,n_ants)
        The distance between an RFI sources and each antenna in metres.

    Returns:
    --------
    delays: array-like (n_time,n_src,n_bl)
        The time delays of the RFI signals between each pair of antennas.
    """

    c = 2.99792458e8
    a1, a2 = np.triu_indices(distances.shape[-1], 1)
    delays = (distances[...,a1]-distances[...,a2])/c

    return delays

def sinc_beam(ang_sep, params):
    """
    A sinc beam that narrows with increasing frequency.
    A parameter $\alpha \propto D$ is present to control
    the width of the beam.

    Paramters:
    ----------
    ang_sep: np.array (n_time,n_ant,n_srcs)
        The radius at which to evaluate the beam function.
        Angular separation, in degrees, from the pointing direction.
    params: array-like (2,)
        Contains alpha, the width parameter, proportional to the
        dish diameter. And the frequency. (alpha, freq)

    Returns:
    --------
    beam: np.array (n_time,n_freqs,n_ant,n_srcs)
        The attenuation due to the beam at the given angular separation.
    """

    alpha, freq = params
    beam = np.sinc(alpha*freq[None,:,None,None]*ang_sep[:,None,:,:,:])

    return beam
