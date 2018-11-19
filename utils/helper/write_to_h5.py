import h5py

def save_input(save_file, target, lm, UVW, A1, A2, spectra, bandpass,
               auto_gains, cross_gains):

    with h5py.File(save_file, 'a') as fp:
        fp['/input/target'] = target
        fp['/input/lm'] = lm
        fp['/input/UVW'] = UVW
        fp['/input/A1'] = A1
        fp['/input/A2'] = A2
        fp['/input/spectra'] = spectra
        fp['/input/bandpass'] = bandpass
        fp['/input/auto_pol_gains'] = auto_gains
        fp['/input/cross_pol_gains'] = cross_gains

def save_output(save_file, vis, clean):

    with h5py.File(save_file, 'a') as fp:
        if clean:
            fp['/output/vis_clean'] = vis
        else:
            fp['/output/vis_dirty'] = vis
