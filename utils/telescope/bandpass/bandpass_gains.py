import numpy as np

def bandpass_time_variation(modes, amplitudes, time):
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

    signal /= 10*np.max(np.abs(signal))
    signal += 1

    return signal

def get_bandpass_and_gains(target_flux, obs_times):
    """
    Loads in the bandpasses from measurements and applys radnom time varying
    gains.

    Parameters:
    -----------
    target_flux: float
        The flux of the target used to set the gain variable/sensitivity in the
        correct region.
    obs_times: array
        Observation times as unix timestamps

    Returns:
    --------
    bandpass: array (1, n_ants, freq_chans, n_pols)
        Bandpass including gain values for every antenna, frequency, time and
        polarization.
    antenna_gains_auto: float
        Initial gain value applied to the autopolarizations (HH, VV).
    antenna_gains_cross: float
        Initial gain value applied to the crosspolarizations (HV, VH).
    """

    bandpass_file = 'utils/telescope/bandpass/MeerKAT_Bandpass_HH-HV-VH-VV.npy'
    # Shape is (1, n_ant, n_chan, 4)
    bandpass = np.load(bandpass_file).astype(np.complex128)

    n_ant = bandpass.shape[1]

    gain_var = 14.875/target_flux

    antenna_gains_auto = gain_var*1.*np.random.rand(n_ant) + 0.5
    antenna_gains_cross = gain_var*0.042*np.random.rand(n_ant) + 0.006

    bandpass[:,:,:,[0,3]] *= antenna_gains_auto[None, :, None, None]
    bandpass[:,:,:,[1,2]] *= antenna_gains_cross[None, :, None, None]

    gain_drift = np.array([bandpass_time_variation(modes=np.random.random(20),
                                 amplitudes=np.random.randn(20, 2),
                                 time=obs_times) for i in range(n_ant)])

    bandpass = bandpass*gain_drift.T[:,:,None,None]

    return bandpass, antenna_gains_auto, antenna_gains_cross
