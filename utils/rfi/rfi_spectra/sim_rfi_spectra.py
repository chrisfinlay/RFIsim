import numpy as np

def wrap_spectra(spectra, n_rfi):

    n_spectra, channels = spectra.shape
    wraps = n_rfi/n_spectra+1
    wrapped_spectra = np.array([spectra for i in range(wraps)])
    wrapped_spectra = wrapped_spectra.reshape(-1, channels)
    rfi_spectrum = wrapped_spectra[:n_rfi]
    rfi_spectrum *= 10**(2*np.random.rand(n_rfi)+5)[:,None]
    # rfi_spectrum *= (1e7*np.random.rand(n_rfi)+1e6)[:,None]

    return rfi_spectrum

def rfi_dist(n_rfi, channels=150, n_chan=4096):

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

    stokes = np.random.randint(1, 4, size=n_rfi)
    signs = (-1)**np.random.randint(0, 2, size=n_rfi)
    stokes_rfi = np.zeros((n_rfi, 4), dtype=np.float64)
    stokes_rfi[:, 0] = 1.0

    for i in range(n_rfi):
        stokes_rfi[i, stokes[i]] = signs[i]

    return stokes_rfi[:,None,None,:]

def get_rfi_spectra(n_chan, n_rfi, n_time):

    file_path = 'utils/rfi/rfi_spectra/RFI_Frequency_Spectra.npy'
    spectra = np.load(file_path)
    perm = np.random.permutation(len(spectra))
    spectra[perm] = spectra

    rfi_spectrum = wrap_spectra(spectra, n_rfi)
    freq_i, freq_f = rfi_dist(n_rfi)

    spectrogram = np.zeros((n_rfi, n_time, n_chan, 1), dtype=np.float64)

    for i in range(n_rfi):
        spectrogram[i, 0, freq_i[i]:freq_f[i], 0] = rfi_spectrum[i]

    stokes = rfi_stokes(n_rfi)

    return spectrogram*stokes
