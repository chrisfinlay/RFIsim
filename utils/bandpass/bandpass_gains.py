import numpy as np

def get_bandpass_and_gains(file_path='utils/bandpass/MeerKAT_Bandpass_HH-HV-VH-VV.npy'):

    bandpass_file = file_path
    # Shape is (1, n_ant, n_chan, 4)
    bandpass = np.load(bandpass_file).astype(np.complex128)

    n_ant = bandpass.shape[0]

    antenna_gains_auto = 200*np.random.rand(n_ant) + 50
    antenna_gains_cross = 70*np.random.rand(n_ant) + 10

    bandpass[:,:,:,[0,3]] *= antenna_gains_auto[None, :, None, None]
    bandpass[:,:,:,[1,2]] *= antenna_gains_cross[None, :, None, None]

    return bandpass, antenna_gains_auto, antenna_gains_cross