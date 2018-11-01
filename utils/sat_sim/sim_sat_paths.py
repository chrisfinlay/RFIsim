import datetime
import ephem
import numpy as np
from glob import glob
from pyrap.measures import measures

# Convert RA and DEC coordinates to l and m given a phase centre
def radec_to_lm(ra, dec, phase_center):
    """
    Convert right-ascension and declination to direction cosines.
    Args:
        ra (float):
            Right-ascension in radians.
        dec (float):
            Declination in radians.
        phase_center (np.ndarray):
            The coordinates of the phase center.
    Returns:
        tuple:
            l and m coordinates.
    """

    delta_ra = ra - phase_center[0]
    dec_0 = phase_center[1]

    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec_0) - \
        np.cos(dec) * np.sin(dec_0) * np.cos(delta_ra)

    return l, m

# Extract TLEs from TLE files
def read_tles(tle_dir='utils/sat_sim/TLEs/'):

    tlefiles = glob(tle_dir+'*.txt')
    tles = []
    for tlefile in tlefiles:
        tles += [line.rstrip() for line in open(tlefile)]
    sats = [tles[3*i:(3*i+3)] for i in range(len(tles)//3)]

    return sats

# Get visible satellites
def visible_sats(sats, ska, phase_centre):

    idx_vis = []
    for i, sat in enumerate(sats):
        sat = ephem.readtle(*sat)
        sat.compute(ska)
        l, m = radec_to_lm(sat.ra, sat.dec, phase_centre)
        theta = np.sqrt(l**2 + m**2)
        if sat.alt>0 and theta<np.deg2rad(30):
            idx_vis.append(i)

    return np.array(idx_vis)

# Get l,m tracks for satellites (track time steps and integration time needed as well as visible satellite index)
def get_lm_tracks(target_ra, target_dec, transit, tracking_hours, integration_secs=8):

    # Define the phase centre
    phase_centre = np.deg2rad([target_ra, target_dec])

    # transit time and tracking_hours needed
    start_time = datetime.datetime.strptime(transit, '%a %b %d %H:%M:%S %Y') - \
                 datetime.timedelta(seconds=3600*tracking_hours/2)

    # Set observer location and time
    ska = ephem.Observer()
    ska.epoch = ephem.J2000
    ska.lon = np.rad2deg(measures().observatory('meerkat')['m0']['value'])
    ska.lat = np.rad2deg(measures().observatory('meerkat')['m1']['value'])
    ska.date = start_time.strftime('%Y/%m/%d %H:%M:%S')

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



