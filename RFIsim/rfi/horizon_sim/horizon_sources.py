import numpy as np
import ephem
import datetime
import sys
sys.path.insert(0, '../../..')
from utils.rfi.sat_sim.sim_sat_paths import radec_to_lm, set_observer
from utils.helper.parallelize import parmap

locations = {'ska': (-30.712586, 21.442888),
             'canarvon': (-30.963501, 22.139584),
             'vanwyksvlei': (-30.332197, 21.818244),
             'brandvlei': (-30.472060, 20.465047),
             'williston': (-31.343608, 20.926702)}


def get_bearing(lat1, lon1, lat2, lon2):
    """
    Straight line bearing from 'lat1, lon1' to 'lat2, lon2'.

    Parameters
    ----------
    lat1 : float
        Latitude of the initial position in degrees or radians.
    lon1 : float
        Longitude of the initial position in degrees or radians.
    lat2 : float
        Latitude of the final position in degrees or radians.
    lon2 : float
        Longitude of the final position in degrees or radians.

    Returns
    -------
    bearing : float
        Straight line bearing from initial position pointing to final position.
        Units must be consitent between 'lat1, lon1, lat2, lon2'.
    """
    x, y = lon2-lon1, lat2-lat1
    bearing = np.arctan2(y, x)
    if bearing<0:
        bearing += 2*np.pi

    return bearing

def get_lm(args):
    """
    Get the direction cosine for a bearing on the horizon.
    """
    bearings, obs_date, phase_centre = args
    obs = set_observer(obs_date)
    lm = np.zeros((len(bearings), 2))
    for i, bearing in enumerate(bearings):
        ra, dec = np.rad2deg(obs.radec_of(bearing, 0))
        delta_ra = np.abs(ra - phase_centre[0])
        delta_dec = np.abs(dec - phase_centre[1])
        if delta_ra<90 and delta_dec<90:
            lm[i] = radec_to_lm(ra, dec, phase_centre)
        else:
            lm[i] = -0.7, -0.7

    return lm

def get_horizon_lm_tracks(phase_centre, transit, tracking_hours,
                          integration_secs=8):
    """
    Get the direction cosine tracks for a source on the horizon.
    """
    start_time = datetime.datetime.strptime(transit, '%a %b %d %H:%M:%S %Y') - \
                 datetime.timedelta(seconds=3600*tracking_hours/2)


    time_steps = int(3600*tracking_hours/integration_secs)

    ska = locations['ska']
    towns = ['canarvon', 'vanwyksvlei', 'brandvlei', 'williston']
    bearings = [get_bearing(ska[0], ska[1], *locations[town]) for town in towns]
    # bearings = np.deg2rad(np.arange(360))

    # Set up argument list for parallelization
    obs_times = [start_time + datetime.timedelta(seconds=i*8) for i in range(time_steps)]
    all_rfi = [bearings for i in range(len(obs_times))]
    centres = [phase_centre for i in range(len(obs_times))]
    arg_list = [list(i) for i in zip(all_rfi, obs_times, centres)]

    # Call parallelization function to get l,m and altitude for all time steps
    all_time = np.array(parmap(get_lm, arg_list, proc_power=0.8))

    return all_time
