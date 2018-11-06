import datetime
import ephem
import numpy as np
from glob import glob
from pyrap.measures import measures

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

# Get visible satellites
def visible_sats(sats, obs, phase_centre, beam_radius=30):
    """
    
    sats :         List of lists. Each internal list has the 3 lines of a TLEs as its elements.
    obs  :         PyEphem observer object that has been initialised with a location and date.
    phase_centre : The RA and DEC of the phase centre (central pointing direction) in degrees.
    beam_radius  : Radius the beam is defined out to in degrees.
    
    Returns:
        idx_vis  : List of indices of sats list for which the satellite is visible.
    """
    
    idx_vis = []
    for i, sat in enumerate(sats):
        sat = ephem.readtle(*sat)
        sat.compute(obs)
        l, m = radec_to_lm(sat.ra, sat.dec, phase_centre)
        theta = np.sqrt(l**2 + m**2)
        if sat.alt>0 and theta<np.deg2rad(beam_radius):
            idx_vis.append(i)

    return np.array(idx_vis)

# Get l,m tracks for satellites (track time steps and integration time needed as well as visible satellite index)
def get_lm_tracks(target_ra, target_dec, transit, tracking_hours, integration_secs=8):

    # Define the phase centre
    phase_centre = [target_ra, target_dec]

    # transit time and tracking_hours needed
    start_time = datetime.datetime.strptime(transit, '%a %b %d %H:%M:%S %Y') - \
                 datetime.timedelta(seconds=3600*tracking_hours/2)

    # Set observer location and time
    ska = set_observer(start_time, telescope='meerkat')
#     ska = ephem.Observer()
#     ska.epoch = ephem.J2000
#     ska.lon = np.rad2deg(measures().observatory('meerkat')['m0']['value'])
#     ska.lat = np.rad2deg(measures().observatory('meerkat')['m1']['value'])
#     ska.date = start_time.strftime('%Y/%m/%d %H:%M:%S')

    sats = read_tles()
    idx_vis = visible_sats(sats, ska, phase_centre)

    time_steps = int(3600*tracking_hours/integration_secs)
    lm = np.zeros((time_steps, 2, 2))
    vis_sats = [ephem.readtle(*sats[idx_vis[i]]) for i in range(2)]
    for i in range(len(lm)):
        ska.date += ephem.second*integration_secs
        for j, sat in enumerate(vis_sats):
            sat.compute(ska)
            lm[i, j] = radec_to_lm(sat.ra, sat.dec, phase_centre)

    return lm



