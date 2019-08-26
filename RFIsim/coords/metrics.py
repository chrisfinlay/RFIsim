import numpy as np

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

    return np.rad2deg(theta)
