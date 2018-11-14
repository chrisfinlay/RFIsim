import numpy as np
import pandas as pd


def inview(phase_centre, radius, min_flux):
    """
    Search NVSS and SUMSS catalogues for radio sources within the FoV and
    brighter than a minimum flux.

    Parameters
    ----------
    phase_centre : array_like
        Right ascension and and declination of the target direction in degrees.
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
    df = pd.read_csv('utils/catalogues/SUMSS_NVSS_Clean.csv')
    r = np.sqrt((df['RA']-phase_centre[0])**2 + (df['DEC']-phase_centre[1])**2)
    df = df[(r<radius) & (df['Flux']>min_flux)]

    return df
