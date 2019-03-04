import datetime
import time as tme
import ephem
import numpy as np
from glob import glob
from pyrap.measures import measures
import sys
sys.path.insert(0, '../../..')
from utils.helper.parallelize import parmap

def radec_to_lm(ra, dec, phase_centre):
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
    (l,m) : tuple
        l and m coordinates. The direction cosines.
    """
    ra, dec = np.deg2rad([ra, dec])
    phase_centre = np.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = np.cos(dec)*np.sin(delta_ra)
    m = np.sin(dec)*np.cos(dec_0) - np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)

    return l, m

def angular_separation(ra, dec, phase_centre):
    """
    Calculates the angular separation between a source and the phase centre.

    Parameters
    ----------
    ra : float
        Right-ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    phase_centre : tuple
        Right-ascension and declination of the phase centre in degrees.

    Returns
    ------
    theta : float
        The angular separation between the phase centre and given source in
        degrees.
    """
    ra1, dec1 = np.deg2rad([ra, dec])
    ra2, dec2 = np.deg2rad(phase_centre)

    theta = np.arccos(np.sin(dec1)*np.sin(dec2) + \
            np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))

    return theta

def read_tles(tle_dir='utils/rfi/sat_sim/TLEs/'):
    """
    Convert TLE text files in a given directory to a list of TLEs.

    Parameters
    ----------
    tle_dir : str
        Path to the directory containing TLE .txt files.

    Returns
    -------
    sats : List of lists.
        Each internal list has the 3 lines of a TLE as its elements.
    """
    tlefiles = glob(tle_dir+'*.txt')
    tles = []
    for tlefile in tlefiles:
        tles += [line.rstrip() for line in open(tlefile)]
    sats = [tles[3*i:(3*i+3)] for i in range(len(tles)//3)]

    return sats

def set_observer(date, telescope='meerkat'):
    """
    Create a PyEphem Observer object.

    Parameters
    ----------
    date : datetime object
        Start date and time of the observation.
    telescope : str
        Telescope name at the location of the observer.

    Returns
    -------
    obs   : PyEphem Observer
        An observer object that can be used to compute astronomical quantities.
    """
    obs = ephem.Observer()
    obs.epoch = ephem.J2000
    obs.lon = np.rad2deg(measures().observatory(telescope)['m0']['value'])
    obs.lat = np.rad2deg(measures().observatory(telescope)['m1']['value'])
    obs.date = date.strftime('%Y/%m/%d %H:%M:%S')

    return obs

def get_lm_and_alt(args):
    """
    Calculate direction cosines and altitudes for satellite TLEs given an
    observation date.

    Parameters
    ----------
    args : tuple
        3 element tuple containing (sats, obs_date, phase_centre).
        sats : list
            3 element lists containing TLE strings.
        obs_date : datetime object
            Observation time and date.
        phase_centre : array_like
            Right ascension and declination of phase centre (pointing centre)
            in degrees.

    Returns
    -------
    lmalt : ndarray
        Array of shape (n_sats, 3) containing the direction cosines (l,m)
        and altitude of satellites in sats.
    """
    sats, obs_date, phase_centre = args
    obs = set_observer(obs_date)
    lmalt = np.zeros((len(sats), 4))
    for i, sat in enumerate(sats):
        sat = ephem.readtle(*sat)
        sat.compute(obs)
        ra, dec = np.rad2deg([sat.ra, sat.dec])
        l, m = radec_to_lm(ra, dec, phase_centre)
        theta = angular_separation(ra, dec, phase_centre)
        lmalt[i] = l, m, sat.alt, theta

    return lmalt

def get_visible_sats(lm_alt, radius=30):
    """
    Get the satellites that are within the FoV and above the horizon.

    Parameters
    ----------
    lm_alt : ndarray
        Array of shape (time_steps, n_sats, 4). It contains all the l,m
        coordinates, altitude and angular separation from phase centre of
        every satellite for every time step.
    radius : float
        Radial field of view in degrees.

    Returns
    -------
    lm_alt_vis : ndarray
        Only satellites that are visible at some time in the observation. If
        satellite is below the horizon l and m are set to -0.7 (edge l,m plane).
    """
    r = np.sqrt(lm_alt[:,:,0]**2+lm_alt[:,:,1]**2)
    visible = ((lm_alt[:,:,2]>0) & (r<np.deg2rad(radius)) &
               (lm_alt[:,:,-1]<60)).astype(int)

    # Satellite must be visible for at least 1 time step
    idx_vis = np.where(np.sum(visible, axis=0)>0)[0]
    lm_alt_vis = lm_alt[:,idx_vis,:]

    # Satellite must be above the horizon
    invis = np.where(lm_alt[:,idx_vis,-1]<0)
    lm_alt_vis[invis[0], invis[1], :2] = -0.7

    return lm_alt_vis[:,:,:2]

def get_lm_tracks(phase_centre, transit, tracking_hours,
                  integration_secs=8):
    """
    Get the l,m tracks for all visible satellites through time.

    Parameters
    ----------
    phase_centre : tuple
        Right ascension and declination of the target source in degrees.
    transit : str
        Time and date when the source is at its highest point (transit).
        Formatted as '%a %b %d %H:%M:%S %Y'. See http://strftime.org/ .
    tracking_hours : float
        The total time the target source will be tracked in hours.
    integration_secs : float
        The integration time for each time step in seconds.

    Returns
    -------
    lm : ndarray
        Array of shape (time_steps, vis_sats, 2) containing the l,m tracks of
        the visible satellites.
    """

    start_time = datetime.datetime.strptime(transit, '%a %b %d %H:%M:%S %Y') - \
                 datetime.timedelta(seconds=3600*tracking_hours/2)

    sats = read_tles()
    time_steps = int(3600*tracking_hours/integration_secs)

    # Set up argument list for parallelization
    obs_times = [start_time + datetime.timedelta(seconds=i*8) for i in range(time_steps)]
    all_sats = [sats for i in range(len(obs_times))]
    centres = [phase_centre for i in range(len(obs_times))]
    arg_list = [list(i) for i in zip(all_sats, obs_times, centres)]

    # Call parallelization function to get l,m and altitude for all time steps
    all_time = np.array(parmap(get_lm_and_alt, arg_list, proc_power=0.8))
    lm = get_visible_sats(all_time)

    obs_times = np.array([tme.mktime(x.timetuple()) for x in obs_times])

    return lm, obs_times
