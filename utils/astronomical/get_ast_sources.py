import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../..')
from utils.rfi.sat_sim.sim_sat_paths import angular_separation


def inview(phase_centre, radius, min_flux):
    """
    Search NVSS and SUMSS catalogues for radio sources within the FoV and
    brighter than a minimum flux.

    Parameters
    ----------
    phase_centre : array_like
        Right ascension and declination of the target direction in degrees.
    radius : float
        Radial field of view about the phase centre to consider in degrees.
    min_flux : float
        Minimum flux of sources to be considered in Jy.

    Returns
    -------
    df : Pandas DataFrame
        Dataframe containing right ascension (deg), declination (deg),
        flux (Jy), major axis (deg), minor axis (deg) and position angle (deg)
        of radio sources fitting the given criteria.
    """
    df = pd.read_csv('utils/astronomical/catalogues/SUMSS_NVSS_Clean.csv')
    theta = angular_separation(df['RA'].values, df['DEC'].values, phase_centre)
    theta = np.rad2deg(theta)
    df = df[(theta<radius) & (df['Flux']>min_flux)]

    return df

def find_closest(ra, dec, min_flux):
    """
    Search NVSS and SUMSS catalogues for the radio source closest to the given
    right ascension and declination with some minimum flux.

    Parameters
    ----------
    ra : float
        Right ascension of the target direction in degrees.
    dec : float
        Declination of the target direction in degrees.
    min_flux : float
        Minimum flux of sources to be considered in Jy.

    Returns
    -------
    new_target : tuple
        The RA and DEC of the closest source with flux greater than 'min_flux'.
    """
    df = pd.read_csv('utils/astronomical/catalogues/SUMSS_NVSS_Clean.csv')
    df = df[df['Flux']>min_flux]
    r = np.sqrt((df['RA']-ra)**2 + (df['DEC']-dec)**2)
    df = df[r==np.min(r)]
    new_target = df['RA'].values[0], df['DEC'].values[0], df['Flux'].values[0]

    return new_target
