import numpy as np

def bandpass_time_variation(modes, amplitudes, times):
    """
    Construct periodic signal from Fourier modes and accompanying amplitudes.

    Parameters:
    -----------
    modes: array (n_modes)
        The frequencies of the modes to add together.
    amplitudes: array (n_modes, 2)
        The amplitudes of each Fourier mode.
    times: array (n_time)
        Time steps to evaluate constructed signal at.

    Returns:
    --------
    signal: array (n_time)
        Fourier constructed signal reduced to a maximum amplitude of 0.1
        sitting on top a constant of 1.
    """

    signal = np.zeros(len(times))
    for i in range(len(modes)):
        signal += amplitudes[i,0]*np.cos(2*np.pi*modes[i]*times) + \
                  amplitudes[i,1]*np.sin(2*np.pi*modes[i]*times)

    signal /= 10*np.max(np.abs(signal))
    signal += 1

    return signal

def get_bandpass_and_gains(config):
    """
    Loads in the bandpasses from measurements and applys radnom time varying
    gains.

    Parameters:
    -----------
    config: dict
        Configuration dictionary created by simulation.

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

    bandpass = np.array([config['telescope']['bandpass_xx'],
                         config['telescope']['bandpass_xy'],
                         config['telescope']['bandpass_yx'],
                         config['telescope']['bandpass_yy']])
    bandpass = np.transpose(bandpass, (1,2,0))[None,...].astype(np.complex128)

    n_ant = bandpass.shape[1]

    gain_var = config['telescope']['gain']*14.875/config['observation']['target_flux']

    antenna_gains_auto = gain_var*1.*np.random.rand(n_ant) + 0.5
    antenna_gains_cross = gain_var*0.042*np.random.rand(n_ant) + 0.006

    bandpass[:,:,:,[0,3]] *= antenna_gains_auto[None, :, None, None]
    bandpass[:,:,:,[1,2]] *= antenna_gains_cross[None, :, None, None]

    max_freq = float(1./config['telescope']['time_var'].seconds)
    times = np.array([time.timestamp() for time in config['observation']['obs_times']])
    gain_drift = np.array([bandpass_time_variation(modes=0.5*max_freq*np.random.random(20)+0.5*max_freq,
                                 amplitudes=np.random.randn(20, 2),
                                 times=times) for i in range(n_ant)])

    bandpass = bandpass*gain_drift.T[:,:,None,None]

    return bandpass, antenna_gains_auto, antenna_gains_cross
