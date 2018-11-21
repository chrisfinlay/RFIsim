import numpy as np

def get_bandpass_and_gains():

    bandpass_file = 'utils/telescope/bandpass/MeerKAT_Bandpass_HH-HV-VH-VV.npy'
    # Shape is (1, n_ant, n_chan, 4)
    bandpass = np.load(bandpass_file).astype(np.complex128)

    n_ant = bandpass.shape[0]

    antenna_gains_auto = 1.2*np.random.rand(n_ant) + 0.3
    antenna_gains_cross = 0.042*np.random.rand(n_ant) + 0.006

    bandpass[:,:,:,[0,3]] *= antenna_gains_auto[None, :, None, None]
    bandpass[:,:,:,[1,2]] *= antenna_gains_cross[None, :, None, None]

    return bandpass, antenna_gains_auto, antenna_gains_cross
