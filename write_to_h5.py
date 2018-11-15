import h5py

# Set file name to save data to

save_file = 'ra=' + str(target_ra) + '_dec=' + str(target_dec) + \
            '_int_secs=' + str(integration_secs) + \
            '_track-time=' + str(round(tracking_hours, 1)) + \
            'hrs_nants=' + str(nant) + '_nchan=' + str(nchan) + '.h5'


# Need to change the number of sources variably
with h5py.File(save_file, 'a') as fp:
    fp['/input/lm_sats'] = lm_sats
    fp['/input/UVW'] = UVW
    fp['/input/A1'] = A1
    fp['/input/A2'] = A2
    fp['/input/bandpass'] = bandpass


with h5py.File(save_file, 'a') as fp:
    fp['/input/spectra'] = spectra


with h5py.File(save_file, 'a') as fp:
    fp['/input/auto_pol_gains'] = antenna_gains_auto
    fp['/input/cross_pol_gains'] = antenna_gains_cross

    # Save output
    if j==0:
        with h5py.File(save_file, 'a') as fp:
            fp['/output/vis_dirty'] = vis

    else:
        with h5py.File(save_file, 'a') as fp:
            fp['/output/vis_clean'] = vis
