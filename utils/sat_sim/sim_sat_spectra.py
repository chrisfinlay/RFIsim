import numpy as np




spectra = np.load('utils/sat_sim/sat_spectra/Satellite_Frequency_Spectra.npy')
perm = np.random.permutation(len(spectra))
spectra[perm] = spectra

n_spectra, rfi_channels = spectra.shape
wraps = n_sats/n_spectra+1
wrapped_spectra = np.array([spectra for i in range(wraps)])
wrapped_spectra = wrapped_spectra.reshape(-1, rfi_channels)
sat_spectrum = wrapped_spectra[:n_sats]
sat_spectrum *= (1e5*np.random.rand(n_sats)+1e4)[:,None]

full_spectrogram = np.zeros((n_sats, 1, nchan, 1))

#### Create rough RFI frequency probability distribution #######################

samples = np.random.randint(0, nchan-rfi_channels, size=(int(1e6)))
rfi_p = np.concatenate((samples[(samples>330) & (samples<400)],
                      samples[(samples>1040) & (samples<1050)],
                      samples[(samples>1320) & (samples<2020)],
                      samples[(samples>3120) & (samples<3520)]))
rfi_p = np.concatenate((rfi_p, np.random.randint(0, nchan-rfi_channels,
                                                 size=len(rfi_p))))
perm = np.random.permutation(len(rfi_p))
rfi_p = rfi_p[perm]

freq_i = rfi_p[:n_sats]
freq_f = freq_i + rfi_channels
for i in range(n_sats):
    full_spectrogram[i, 0, freq_i[i]:freq_f[i], 0] = sat_spectrum[i]

# Stokes parameters for rfi sources (I, Q, U, V)
stokes_sats = np.random.randint(1, 4, size=n_sats)
signs = (-1)**np.random.randint(0, 2, size=n_sats)
lm_stokes_sats = np.zeros((n_sats, 4))
lm_stokes_sats[:, 0] = 1.0
for i in range(n_sats):
    lm_stokes_sats[i, stokes_sats[i]] = signs[i]

# Save input spectra
s = np.asarray(lm_stokes_sats, dtype=np.float64)[:,None,None,:]
spectra = s*full_spectrogram
