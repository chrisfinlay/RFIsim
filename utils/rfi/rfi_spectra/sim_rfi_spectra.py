import numpy as np

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
    wraps = n_rfi/n_spectra+1
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

    samples = np.random.randint(0, n_chan-channels, size=(int(1e6)))
    rfi_p = np.concatenate((samples[(samples>330) & (samples<400)],
                          samples[(samples>1040) & (samples<1050)],
                          samples[(samples>1320) & (samples<2020)],
                          samples[(samples>3120) & (samples<3520)]))
    rfi_p = np.concatenate((rfi_p, np.random.randint(0, n_chan-channels,
                                                     size=len(rfi_p))))
    perm = np.random.permutation(len(rfi_p))
    freq_i = (rfi_p[perm])[:n_rfi]

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

def get_rfi_spectra(n_chan, n_rfi, n_time):
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

    Returns
    -------
    spectrogram: array (n_rfi, n_time, n_chan, 4)
        Polarmetric spectrum for each RFI source at each time step.
    """

    file_path = 'utils/rfi/rfi_spectra/RFI_Frequency_Spectra.npy'
    spectra = np.load(file_path)
    perm = np.random.permutation(len(spectra))
    spectra[perm] = spectra

    rfi_spectrum = wrap_spectra(spectra, n_rfi)
    freq_i, freq_f = rfi_dist(n_rfi)

    spectrogram = np.zeros((n_rfi, n_time, n_chan, 1), dtype=np.float64)

    for i in range(n_rfi):
        spectrogram[i, :, freq_i[i]:freq_f[i], 0] = rfi_spectrum[i, None]

    stokes = rfi_stokes(n_rfi)

    time_dep = np.array([rfi_time_variation(modes=np.random.random(20),
                                 amplitudes=np.random.randn(20, 2),
                                 time=np.arange(n_time)) for i in range(n_rfi)])

    print(time_dep.shape)
    print(time_dep)
    print(stokes.shape)
    print(stokes)
    print(spectrogram.shape)
    print(spectrogram)

    return spectrogram*stokes*time_dep[:,:,None,None]
