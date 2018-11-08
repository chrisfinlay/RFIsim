import datetime
import ephem
import numpy as np
from glob import glob
from pyrap.measures import measures
import sys
sys.path.insert(0, '../..')
from utils.parallelize import parmap

# Convert RA and DEC coordinates to l and m given a phase centre
def radec_to_lm(ra, dec, phase_centre):
    """
    Convert right-ascension and declination to direction cosines.
    Args:
        ra (float):
            Right-ascension in degrees.
        dec (float):
            Declination in degrees.
        phase_center (np.ndarray):
            The coordinates of the phase center.
    Returns:
        tuple:
            l and m coordinates.
    """
    phase_centre = np.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec_0) - \
        np.cos(dec) * np.sin(dec_0) * np.cos(delta_ra)

    return l, m

# Extract TLEs from TLE files
def read_tles(tle_dir='utils/sat_sim/TLEs/'):
    """
    tle_dir : Path to the directory containing TLE .txt files

    Returns :
        sats : List of lists. Each internal list has the 3 lines of a TLEs as its elements.
    """

    tlefiles = glob(tle_dir+'*.txt')
    tles = []
    for tlefile in tlefiles:
        tles += [line.rstrip() for line in open(tlefile)]
    sats = [tles[3*i:(3*i+3)] for i in range(len(tles)//3)]

    return sats

def set_observer(date, telescope='meerkat'):
    """
    date      : Datetime object with the start time of the observation.
    telescope : String giving the telescope name that is used to search a database.

    Returns:
        obs   : PyEphem Observer object
    """

    obs = ephem.Observer()
    obs.epoch = ephem.J2000
    obs.lon = np.rad2deg(measures().observatory(telescope)['m0']['value'])
    obs.lat = np.rad2deg(measures().observatory(telescope)['m1']['value'])
    obs.date = date.strftime('%Y/%m/%d %H:%M:%S')

    return obs

# Get l,m and altitude of every satellite
def get_lm_and_alt(args):

    sats, obs_date, phase_centre = args
    obs = set_observer(obs_date)
    lmalt = np.zeros((len(sats), 3))
    for i, sat in enumerate(sats):
        sat = ephem.readtle(*sat)
        sat.compute(obs)
        l, m = radec_to_lm(sat.ra, sat.dec, phase_centre)
        lmalt[i] = l, m, sat.alt
    return lmalt

# Get visible satellites
def get_visible_sats(lm_alt):
    """
    lm_alt         : Array of shape [time, sats, 3]. It contains all the l,m and altitude of every satellite for every time step.

    Returns :
        lm_alt_vis : Array with only satellites that are visible at some time in the observation.
                     l, m is set to -0.5 (outside of 30 deg beam) if below the horizon.
    """
    r = np.sqrt(lm_alt[:,:,0]**2+lm_alt[:,:,1]**2)
    visible = ((lm_alt[:,:,-1]>0) & (r<np.deg2rad(30))).astype(int)

    idx_vis = np.where(np.sum(visible, axis=0)>0)[0]
    lm_alt_vis = lm_alt[:,idx_vis,:]

    invis = np.where(lm_alt[:,idx_vis,-1]<0)
    lm_alt_vis[invis[0], invis[1], :2] = -0.5

    return lm_alt_vis[:,:,:2]

# Get l,m tracks for satellites (track time steps and integration time needed as well as visible satellite index)
def get_lm_tracks(target_ra, target_dec, transit, tracking_hours, integration_secs=8):

    # Define the phase centre
    phase_centre = [target_ra, target_dec]

    # transit time and tracking_hours needed
    start_time = datetime.datetime.strptime(transit, '%a %b %d %H:%M:%S %Y') - \
                 datetime.timedelta(seconds=3600*tracking_hours/2)

    # Set observer location and time
    ska = set_observer(start_time, telescope='meerkat')

    # Read TLE files
    sats = read_tles()

    # Set arguments for l,m and altitude calls
    time_steps = int(3600*tracking_hours/integration_secs)

    obs_times = [start_time + datetime.timedelta(seconds=i*8) for i in range(time_steps)]
    all_obs = [set_observer(obs_times[i]) for i in range(len(obs_times))]
    all_sats = [sats for i in range(len(obs_times))]
    centres = [phase_centre for i in range(len(obs_times))]
    arg_list = [list(i) for i in zip(all_sats, obs_times, centres)]

    # Call parallelization function to get l,m and altitude for all time steps
    all_time = np.array(parmap(get_lm_and_alt, arg_list, proc_power=0.8))

    # Get visible satellite tracks lm is shape (time_steps, vis_sats, 2)
    lm = get_visible_sats(all_time)

    return lm
