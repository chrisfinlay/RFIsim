import numpy as np

def wrap_spectra(spectra, n_sats):

    n_spectra, channels = spectra.shape
    wraps = n_sats/n_spectra+1
    wrapped_spectra = np.array([spectra for i in range(wraps)])
    wrapped_spectra = wrapped_spectra.reshape(-1, channels)
    sat_spectrum = wrapped_spectra[:n_sats]
    sat_spectrum *= (1e5*np.random.rand(n_sats)+1e4)[:,None]

    return sat_spectrum

def rfi_dist(n_sats, channels=150, n_chan=4096):

    samples = np.random.randint(0, n_chan-channels, size=(int(1e6)))
    rfi_p = np.concatenate((samples[(samples>330) & (samples<400)],
                          samples[(samples>1040) & (samples<1050)],
                          samples[(samples>1320) & (samples<2020)],
                          samples[(samples>3120) & (samples<3520)]))
    rfi_p = np.concatenate((rfi_p, np.random.randint(0, n_chan-channels,
                                                     size=len(rfi_p))))
    perm = np.random.permutation(len(rfi_p))
    freq_i = (rfi_p[perm])[:n_sats]

    return freq_i, freq_i+channels

def sat_stokes(n_sats):

    stokes = np.random.randint(1, 4, size=n_sats)
    signs = (-1)**np.random.randint(0, 2, size=n_sats)
    stokes_sats = np.zeros((n_sats+1, 4), dtype=np.float64)
    stokes_sats[:, 0] = 1.0

    for i in range(n_sats):
        stokes_sats[i+1, stokes[i]] = signs[i]

    return stokes_sats[:,None,None,:]

def get_sat_spectra(n_chan, n_sats, n_time):

    file_path = 'utils/rfi/sat_sim/sat_spectra/Satellite_Frequency_Spectra.npy'
    spectra = np.load(file_path)
    perm = np.random.permutation(len(spectra))
    spectra[perm] = spectra

    sat_spectrum = wrap_spectra(spectra, n_sats)
    freq_i, freq_f = rfi_dist(n_sats)

    spectrogram = np.zeros((n_sats+1, n_time, n_chan, 1), dtype=np.float64)
    spectrogram[0,:,:,0] = 1e-9

    for i in range(n_sats):
        spectrogram[i+1, 0, freq_i[i]:freq_f[i], 0] = sat_spectrum[i]

    stokes = sat_stokes(n_sats)

    return spectrogram*stokes
