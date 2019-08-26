#import sys
#sys.path.insert(0, '../..')
import numpy as np
from RFIsim.coords.transforms import radec_to_lmn

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
    a = 3e-2
    b = 2*np.pi/123
    phi = 320
    beam = np.exp(-1j*(freq[None,:,None,None]*b - phi))*np.sinc(a*freq[None,:,None,None]*ang_sep[:,None,:,:])
#     beam = np.sinc(alpha*freq[None,:,None,None]*ang_sep[:,None,:,:])

    return beam

def auto_beam(ra, dec, freqs, target):
    """
    Generate the complex beam cube for the auto-polarizations (HH, VV).

    Parameters
    ----------
    ra : np.array (n_time, n_ant, n_src)
        Right ascension of sources in decimal degrees.
    dec : np.array (n_time, n_ant, n_src)
        Declination of sources in decimal degrees
    freqs : np.array (n_freq,)
        Frequencies in Hertz.

    Returns
    -------
    beam : np.array (n_time, n_freq, n_ant, n_src)
        Complex values of the beam sensitivity.
    """
    l, m, _ = radec_to_lmn(ra, dec, target)
    l = l[:,None,:,:]
    m = m[:,None,:,:]
    freqs = freqs[None,:,None,None]
    a = 3e-2
    b = 2*np.pi/123
    phi = 320
    phi = np.deg2rad(phi)
    r = np.sqrt(l**2+m**2)
    beam = np.exp(-1j*(freqs*b - phi))*np.sinc(a*r*freqs)

    return beam

def cross_beam(ra, dec, freqs, target):
    """
    Generate the complex beam cube for the cross-polarizations (HV, VH).

    Parameters
    ----------
    ra : np.array (n_time, n_ant, n_src)
        Right ascension of sources in decimal degrees.
    dec : np.array (n_time, n_ant, n_src)
        Declination of sources in decimal degrees
    freqs : np.array (n_freq,)
        Frequencies in Hertz.

    Returns
    -------
    beam : np.array (n_time, n_freq, n_ant, n_src)
        Complex values of the beam sensitivity.
    """
    l, m, _ = radec_to_lmn(ra, dec, target)
    l = l[:,None,:,:]
    m = m[:,None,:,:]
    freqs = freqs[None,:,None,None]
    a = 2.5e-2
    b = 3e-2
    r = np.sqrt(l**2+m**2)
    lam = 2*np.pi/123
    phi = np.deg2rad(140-180)
    freq_dep = np.exp(-1j*(lam*freqs-phi))
    beam = -l*m*np.sinc(a*r*freqs)*np.exp(-(r/b)**2)*freq_dep

    return beam
