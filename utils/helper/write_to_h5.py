import h5py

def save_input(save_file, target, astro_srcs, rfi_lm, UVW, A1, A2, rfi_spectra,
               bandpass, freqs, auto_gains, cross_gains, obs_times):

    with h5py.File(save_file, 'a') as fp:
        fp['/input/target'] = target
        fp['/input/astro_sources'] = astro_srcs
        fp['/input/astro_sources_headings'] = astro_srcs.keys().values.astype('|S8')
        fp['/input/rfi_lm'] = rfi_lm
        fp['/input/rfi_spectra'] = rfi_spectra
        fp['/input/UVW'] = UVW
        fp['/input/A1'] = A1
        fp['/input/A2'] = A2
        fp['/input/frequencies'] = freqs
        fp['/input/bandpass'] = bandpass.real
        fp['/input/auto_pol_gains'] = auto_gains
        fp['/input/cross_pol_gains'] = cross_gains
        fp['/input/unix_times'] = obs_times

def save_output(save_file, vis, clean):

    with h5py.File(save_file, 'a') as fp:
        if clean:
            fp['/output/vis_clean'] = vis
        else:
            fp['/output/vis_dirty'] = vis
