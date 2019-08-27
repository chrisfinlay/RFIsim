from skyfield.api import Topos, load
from RFIsim.coords.transforms import radec_to_lmn
import numpy as np

def delay_correction(config):
    """
    Calculate the delays per antenna and time
    to get from zenith as the pointing centre to
    the actual phase centre.

    Parameters:
    -----------
    config: dict
        The configuration dictionary created by RFIsim.load_config

    Returns:
    --------
    distances:  np.array (n_time, n_ant)
        The distances in metres for the correction.
    """

    MeerKAT_gps = config['telescope']['GPS_coords']
    times = config['observation']['obs_times']
    enu = config['telescope']['ENU_coords']
    ra, dec = config['observation']['target']
    zenith = Topos(latitude_degrees=MeerKAT_gps[0],
                   longitude_degrees=MeerKAT_gps[1],
                   elevation_m=1e15)
    ts = load.timescale()
    t = ts.utc(times)
    lmn = np.array(radec_to_lmn(ra, dec,
                                [zenith.at(t).radec()[0]._degrees,
                                 zenith.at(t).radec()[1]._degrees]))
    distances = np.dot(enu, lmn).T

    return distances
