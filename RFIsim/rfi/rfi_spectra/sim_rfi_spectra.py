import numpy as np

def pulse(x, A, m, s, n):
    """
    Create a sersic profile pulse.

    $A\exp{-0.5(|x-m|/|s|)^n}$

    Parameters
    ----------
    x: array
        Domain over which to evaluate the pulse.
    A: float
        Amplitude of the pulse.
    m: float
        Position of the pulse in the domain.
    s: float
        Width of the pulse.
    n: float
        Sersic index of the pulse. e.g. n=2 corresponds to a gaussian

    Returns
    -------
    pulse: array
        Values of the pulse at domain sample points.
    """
    return A*np.exp(-0.5*np.abs(((x-m)/s))**n)

def random_spectra(x, s, n):
    """
    Create a random signal composed of 2-5 random pulses and noise.

    Parameters
    ----------
    x: array
        Domain over which to evaluate signal.
    s: float
        Maximum pulse width.
    n: float
        Sersic index.

    Returns
    -------
    signal: array
        Constructed random signal.
    """

    centre_band = int(0.6*len(x))
    edge = int(0.2*len(x))

    s = np.sum([pulse(x, 0.7+0.6*np.random.random(),
                      centre_band*np.random.random()+edge,
                      s*np.random.random(), n)
                for _ in range(np.random.randint(2, 6))], axis=0)

    s += 0.03*np.random.randn(len(x))
    s -= np.min(s)
    s *= pulse(x, 1, np.mean(x), (x[-1]-x[0])/3., 10)

    return s

def rfi_time_variation(modes, amplitudes, time):
    """
    Construct periodic signal from Fourier modes and accompanying amplitudes.

    Parameters:
    -----------
    modes: array (n_modes)
        The frequencies of the modes to add together.
    amplitudes: array (n_modes, 2)
        The amplitudes of each Fourier mode.
    time: array
        Time steps to evaluate constructed signal at.

    Returns:
    --------
    signal: array (n_time)
        Fourier constructed signal reduced to a maximum amplitude of 0.1
        sitting on top a constant of 1.
    """

    signal = np.zeros(len(time))
    for i in range(len(modes)):
        signal += amplitudes[i,0]*np.cos(2*np.pi*modes[i]*time) + \
                  amplitudes[i,1]*np.sin(2*np.pi*modes[i]*time)

    signal /= 2*np.max(np.abs(signal))
    signal += 0.5

    return signal

def wrap_spectra(spectra, n_rfi):
    """
    Concatenate predefined spectra so as to have enough for all RFI sources.

    Parameters
    ----------
    spectra: array (n_spectra, n_chan)
        Spectra to assign to RFI sources
    n_rfi: int
        Number of RFI sources to assign a spectrum to.

    Returns
    -------
    rfi_spectrum: array (n_rfi, n_chan)
        Spectrum for each RFI source.
    """

    n_spectra, channels = spectra.shape
    wraps = n_rfi//n_spectra+1
    wrapped_spectra = np.array([spectra for i in range(wraps)])
    wrapped_spectra = wrapped_spectra.reshape(-1, channels)
    rfi_spectrum = wrapped_spectra[:n_rfi]
    rfi_spectrum *= 10**(3*np.random.rand(n_rfi)+4)[:,None]

    return rfi_spectrum

def rfi_dist(n_rfi, channels=150, n_chan=4096):
    """
    Sample frequency positions for RFI sources from a defined distribution over
    frequency.

    Parameters
    ----------
    n_rfi: int
        Number of RFI sources.
    channels: int
        Bandwidth of the RFI in number of frequency channels.
    n_chan: int
        Number of channels in the entire frequency range.

    Returns
    -------
    freq_i: array (n_rfi)
        Initial positions in frequency channel where RFI spectrum/bandwidth
        starts.
    freq_f: array (n_rfi)
        Final positions in frequency channel where RFI spectrum/bandwidth
        ends.
    """

    samples = np.load('utils/rfi/rfi_spectra/MeerKAT_RFI_Prob_Frequency.npy')
    samples -= int(channels/2)
    samples = samples[(samples>0) & (samples<n_chan-channels)]
    samples = samples[np.random.permutation(len(samples))]

    freq_i = samples[:n_rfi]

    return freq_i, freq_i+channels

def rfi_stokes(n_rfi):
    """
    Randomly assign a polarization to RFI sources.

    Parameters
    ----------
    n_rfi: int
        Number of RFI sources to assign a polarization to.

    Returns
    -------
    stokes_rfi: array (n_rfi, 1, 1, 4)
        Array with 1 in stokes I position and a 1/-1 in Q, U or V position.
    """

    stokes = np.random.randint(1, 4, size=n_rfi)
    signs = (-1)**np.random.randint(0, 2, size=n_rfi)
    stokes_rfi = np.zeros((n_rfi, 4), dtype=np.float64)
    stokes_rfi[:, 0] = 1.0

    for i in range(n_rfi):
        stokes_rfi[i, stokes[i]] = signs[i]

    return stokes_rfi[:,None,None,:]

def get_rfi_spectra(n_chan, n_rfi, n_time, type='s'):
    """
    Assign spectra to RFI sources.

    Parameters
    ----------
    n_chan: int
        Number of frequency channels.
    n_rfi: int
        Number of RFI sources.
    n_time: int
        Number of time steps.
    type: char
        Type of RFI spectra to use.
        s := sersic profiles
        g := gaussian profiles
        r := real profiles

    Returns
    -------
    spectrogram: array (n_rfi, n_time, n_chan, 4)
        Polarmetric spectrum for each RFI source at each time step.
    """

    type = type.lower()[0]

    x = np.arange(100)

    if type=='s':
        spectra = np.array([random_spectra(x, s=3., n=4*np.random.random()+1)
                            for _ in range(n_rfi)])
    elif type=='g':
        spectra = np.array([random_spectra(x, s=2., n=2) for _ in range(n_rfi)])
    else:
        file_path = 'utils/rfi/rfi_spectra/RFI_Frequency_Spectra_real.npy'
        spectra = np.load(file_path)
        perm = np.random.permutation(len(spectra))
        spectra[perm] = spectra

    rfi_spectrum = wrap_spectra(spectra, n_rfi)

    freq_i, freq_f = rfi_dist(n_rfi, channels=rfi_spectrum.shape[1])

    spectrogram = np.zeros((n_rfi, n_time, n_chan, 1), dtype=np.float64)

    for i in range(n_rfi):
        spectrogram[i, :, freq_i[i]:freq_f[i], 0] = rfi_spectrum[i, None]

    stokes = rfi_stokes(n_rfi)

    time_dep = np.array([rfi_time_variation(modes=np.random.random(20),
                                 amplitudes=np.random.randn(20, 2),
                                 time=np.arange(n_time)) for i in range(n_rfi)])

    spectrogram = spectrogram*stokes
    spectrogram = spectrogram*time_dep[:,:,None,None]

    return spectrogram
