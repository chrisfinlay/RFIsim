import astropy.io.fits as fits
import numpy as np

def gen_domain(resolution=513, channels=100):

    l = np.linspace(-30, 30, resolution)
    m = np.linspace(-30, 30, resolution)
    f = np.linspace(800, 1800, channels)

    domain_l = np.deg2rad(np.ones((len(f), len(l), len(m)))*l[None, :,None])
    domain_m = np.deg2rad(np.ones((len(f), len(l), len(m)))*m[None, None, :])
    domain_f = np.ones((len(f), len(l), len(m)))*f[:, None, None]

    return domain_l, domain_m, domain_f

def auto_beam(ll, mm, ff, a=3e-2, b=2*np.pi/123, phi=np.deg2rad(140+180)):

    rr = np.sqrt(ll**2+mm**2)

    return np.exp(-1j*(ff*b - phi))*np.sinc(a*rr*ff)

def cross_beam(ll, mm, ff, a=3e-2, b=2.5e-2, sig=[1.2e-1, 1e-3, 1410, 4e-4]):

    rr = np.sqrt(ll**2+mm**2)
    lam = 2*np.pi/123
    phi = np.deg2rad(140-180)
    freq_dep = np.exp(-1j*(lam*ff-phi))*(sig[1]/(1+np.exp(-(sig[0]*(ff-sig[2])))) + sig[3])

    return -ll*mm*np.sinc(b*rr*ff)*np.exp(-(rr/a)**2)*freq_dep

def set_header(comp, hdul):

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
            
                print('Creating FITS file number {} of 8'.format(i*2**2 + j*2**1 + k*2**0 + 1))
        
                hdu = fits.PrimaryHDU(part)
                hdul = fits.HDUList([hdu])
                hdul = set_header(comp, hdul)

                hdul.writeto('beams/FAKE_'+pols[2*i+k]+'_'+parts[j]+'.fits', overwrite=True)
                print('Finished FITS file number {} of 8\n'.format(i*2**2 + j*2**1 + k*2**0 + 1))

if __name__ == '__main__':

    beam_sim_and_write_fits()
