import astropy.io.fits as fits
import numpy as np

def gen_domain(resolution=513, channels=100):
    """
    Generate the domain over which the beam is evaluated.

    Parameters
    ----------
    resolution : int
        The number of points on the l, m axes.
    channels : int
        The number of point on the frequency axis.

    Returns
    -------
    domain_l : ndarray
        Array of shape (channels, resolution, resolution) containing the l
        coordinate points varying along axis 1.
    domain_m : ndarray
        Array of shape (channels, resolution, resolution) containing the m
        coordinate points varying along axis 2.
    domain_f : ndarray
        Array of shape (channels, resolution, resolution) containing the
        frequency coordinate points varying along axis 0.
    """
    l = np.linspace(-30, 30, resolution)
    m = np.linspace(-30, 30, resolution)
    f = np.linspace(800, 1800, channels)

    domain_l = np.deg2rad(np.ones((len(f), len(l), len(m)))*l[None, :,None])
    domain_m = np.deg2rad(np.ones((len(f), len(l), len(m)))*m[None, None, :])
    domain_f = np.ones((len(f), len(l), len(m)))*f[:, None, None]

    return domain_l, domain_m, domain_f

def auto_beam(ll, mm, ff, a=3e-2, b=2*np.pi/123, phi=320):
    """
    Generate the complex beam cube for the auto-polarizations (HH, VV).

    Parameters
    ----------
    ll : ndarray
        Array of shape (channels, resolution, resolution) containing the l
        coordinate points varying along axis 1.
    mm : ndarray
        Array of shape (channels, resolution, resolution) containing the m
        coordinate points varying along axis 2.
    ff : ndarray
        Array of shape (channels, resolution, resolution) containing the
        frequency coordinate points varying along axis 0.
    a : float
        Width of the sinc component of the beam.
    b : float
        Oscillatory frequency of the complex exponential component.
    phi : float
        Phase offset in the complex exponential component in degrees.

    Returns
    -------
    beam : ndarray
        Array of shape (channels, resolution, resolution) containing the
        complex values of the beam sensitivity.
    """
    phi = np.deg2rad(phi)
    rr = np.sqrt(ll**2+mm**2)
    beam = np.exp(-1j*(ff*b - phi))*np.sinc(a*rr*ff)

    return beam

def cross_beam(ll, mm, ff, a=2.5e-2, b=3e-2):
    """
    Generate the complex beam cube for the cross-polarizations (HV, VH).

    Parameters
    ----------
    ll : ndarray
        Array of shape (channels, resolution, resolution) containing the l
        coordinate points varying along axis 1.
    mm : ndarray
        Array of shape (channels, resolution, resolution) containing the m
        coordinate points varying along axis 2.
    ff : ndarray
        Array of shape (channels, resolution, resolution) containing the
        frequency coordinate points varying along axis 0.
    a : float
        Width of the sinc component of the beam.
    b : float
        Characteristic distance of the exponential dropoff component.

    Returns
    -------
    beam : ndarray
        Array of shape (channels, resolution, resolution) containing the
        complex values of the beam sensitivity.
    """
    rr = np.sqrt(ll**2+mm**2)
    lam = 2*np.pi/123
    phi = np.deg2rad(140-180)
    # sig = [1.2e-1, 1e-3, 1410, 4e-4]
    # sigmoid = (sig[1]/(1+np.exp(-(sig[0]*(ff-sig[2])))) + sig[3])
    freq_dep = np.exp(-1j*(lam*ff-phi))#*sigmoid
    beam = -ll*mm*np.sinc(a*rr*ff)*np.exp(-(rr/b)**2)*freq_dep

    return beam

def set_header(comp, hdul):
    """
    Set the header of the beam FITS file.

    Parameters
    ----------
    comp : ndarray
        Array of shape (channels, resolution, resolution) containing the
        real or imaginary values of the beam sensitivity.
    hdul : HDUList object
        HDUList of a FITS file object.

    Returns
    -------
    hdul : HDUList object
        The HDUList object with its header defined.
    """

    hdul[0].header['CTYPE1'] = 'L'
    hdul[0].header['CRPIX1'] = comp.shape[1]/2 + 1
    hdul[0].header['CRVAL1'] = 0.0
    hdul[0].header['CDELT1'] = 30.0/comp.shape[1]
    hdul[0].header['CUNIT1'] = 'deg'

    hdul[0].header['CTYPE2']  = 'M'
    hdul[0].header['CRPIX2'] = comp.shape[1]/2 + 1
    hdul[0].header['CRVAL2'] = 0.0
    hdul[0].header['CDELT2'] = 30.0/comp.shape[1]
    hdul[0].header['CUNIT2'] = 'deg'

    hdul[0].header['CTYPE3'] = 'FREQ'
    hdul[0].header['CRPIX3'] = 1
    hdul[0].header['CRVAL3'] = 800000000.0
    hdul[0].header['CDELT3'] = 1000000000.0/comp.shape[0]
    hdul[0].header['CUNIT3'] = 'Hz'

    hdul[0].header['TELESCOP'] = 'FAKE MeerKAT'

    return hdul

def beam_sim_and_write_fits(resolution=513, channels=100):
    """
    Generate a complex beam cube for all 4 Jones terms and write to
    8 separate FITS files.

    Parameters
    ----------
    resolution : int
        The number of points on the l, m axes.
    channels : int
        The number of point on the frequency axis.
    """
    ll, mm, ff = gen_domain(resolution, channels)

    print('\nCreating Auto Beam')
    auto = auto_beam(ll, mm, ff)
    print('Finished Creating Auto Beam')

    print('\nCreating Cross Beam')
    cross = cross_beam(ll, mm, ff)
    print('Finished Creating Cross Beam\n')

    pols = ['xx', 'yy', 'xy', 'yx']

    parts = ['re', 'im']

    for i, comp in enumerate([auto, cross]):
        for j, part in enumerate([comp.real, comp.imag]):
            for k in range(2):

                file_n = i*2**2 + j*2**1 + k*2**0 + 1
                print('Creating FITS file number {} of 8'.format(file_n))

                hdu = fits.PrimaryHDU(part)
                hdul = fits.HDUList([hdu])
                hdul = set_header(comp, hdul)

                file_name = 'beams/FAKE_'+pols[2*i+k]+'_'+parts[j]+'.fits'
                hdul.writeto(file_name, overwrite=True)
                print('Finished FITS file number {} of 8\n'.format(file_n))

if __name__ == '__main__':

    beam_sim_and_write_fits()
