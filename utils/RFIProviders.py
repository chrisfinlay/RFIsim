import numpy as np

import montblanc
from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider
import montblanc.util as mbu


from utils.rfi.sat_sim.sim_sat_paths import radec_to_lm

######## Source Provider #######################################################

class RFISourceProvider(SourceProvider):
    """
    Supplies data to montblanc via data source methods,
    which have the following signature.

    .. code-block:: python

        def point_lm(self, context)
            ...
    """
    def __init__(self, rfi_run, n_time, n_chan, n_ant, freqs, bandpass,
                 gauss_sources, rfi_spectra, rfi_lm, phase_centre, time_step,
                 A1, A2, UVW):
        self.rfi_run = rfi_run
        self.n_time = int(n_time)
        self.n_chan = n_chan
        self.n_ant = n_ant
        self.n_bl = n_ant*(n_ant-1)/2
        self.freqs = freqs
        self.UVW = UVW
        self.bandpass = bandpass
        self.gauss_sources = gauss_sources
        self.n_gsrcs = len(gauss_sources)
        self.rfi_spectra = rfi_spectra
        self.n_rfi = rfi_spectra.shape[0]
        self.rfi_lm = rfi_lm[time_step]
        self.target = phase_centre
        self.time_step = time_step
        self.A1 = A1
        self.A2 = A2


    def name(self):
        """ Name of Source Provider """
        return self.__class__.__name__

    def updated_dimensions(self):
        """ Inform montblanc about dimension sizes """
        dims = [("ntime", self.n_time),  # Timesteps
                ("nchan", self.n_chan),  # Channels
                ("na", self.n_ant),      # Antenna
                ("nbl", self.n_bl),      # Baselines
                ("ngsrc", self.n_gsrcs)]  # Number of astronomical sources
        if self.rfi_run:
            dims.append(("npsrc", self.n_rfi))
        else:
            dims.append(("npsrc", 0))

        return dims

    def frequency(self, context):
        """ Supply frequencies for each channel """
        # Shape (nchan)
        lc, uc = context.array_extents(context.name)[0]

        return self.freqs[lc:uc]

    def point_lm(self, context):
        """ Supply point source lm coordinates to montblanc """

        # Shape (npsrc, 2)
        (ls, us), _ = context.array_extents(context.name)

        return np.asarray(self.rfi_lm, dtype=context.dtype)[ls:us, :]

    def point_stokes(self, context):
        extents = context.dim_extents('npsrc', 'ntime', 'nchan')
        (lp, up), (lt, ut), (lc, uc) = extents
        # (npsrc, ntime, nchan, 4)

        spec = np.ones((1, self.n_time, 1, 1))
        spec = self.rfi_spectra*spec

        return spec[lp:up, lt:ut, lc:uc, :]

    def gaussian_lm(self, context):
        """ Returns an lm coordinate array to Montblanc. """

        # lm = np.empty(context.shape, context.dtype)
        lm = np.empty((self.n_gsrcs, 2), context.dtype)

        # Get the extents of the time, baseline and chan dimension
        (lg, ug) = context.dim_extents('ngsrc')

        ra = self.gauss_sources['RA']
        dec = self.gauss_sources['DEC']

        lm[:,0], lm[:,1] = radec_to_lm(ra, dec, self.target)

        return lm[lg:ug]

    def gaussian_shape(self, context):
        """ Returns a Gaussian shape array to Montblanc """

        (lg, ug) = context.dim_extents('ngsrc')

        emaj = np.deg2rad(self.gauss_sources['MajAxis'].values)
        emin = np.deg2rad(self.gauss_sources['MinAxis'].values)
        pa = np.deg2rad(self.gauss_sources['PA'].values)

        gauss = np.empty((3, self.n_gsrcs), context.dtype)

        gauss[0,:] = emaj * np.sin(pa)
        gauss[1,:] = emaj * np.cos(pa)
        emaj[emaj == 0.0] = 1.0
        gauss[2,:] = emin / emaj

        return gauss[:, lg:ug]

    def gaussian_stokes(self, context):
        """ Return a stokes parameter array to Montblanc """

        # Get the extents of the time, baseline and chan dimension
        extents = context.dim_extents('ngsrc', 'ntime', 'nchan')
        (lg, ug), (lt, ut), (lc, uc) = extents

        stokes = np.zeros((self.n_gsrcs, self.n_time, self.n_chan, 4),
                           context.dtype)

        stokes[:,:,:,0] = self.gauss_sources['Flux'].values[:,None,None]
        stokes[:,:,:,1:] = 0.0

        return stokes[lg:ug, lt:ut, lc:uc, :]

    def direction_independent_effects(self, context):
        # (ntime, na, nchan, npol)
        extents = context.dim_extents('ntime', 'na', 'nchan', 'npol')
        (lt, ut), (la, ua), (lc, uc), (lp, up) = extents

        bp = np.ones((self.n_time, 1, 1, 1))
        bp = self.bandpass*bp
        return bp[lt:ut, la:ua, lc:uc, lp:up]

    def uvw(self, context):
        """ Supply UVW antenna coordinates to montblanc """

        # Shape (ntime, na, 3)
        (lt, ut), (la, ua), (l, u) = context.array_extents(context.name)

        if self.rfi_run:
            idx = np.arange(self.time_step*self.n_bl,
                           (self.time_step+1)*self.n_bl)
            auvw = mbu.antenna_uvw(self.UVW[idx], self.A1, self.A2,
                                   np.array([self.n_bl,]),
                                   nr_of_antenna=self.n_ant)
        else:
            AA1 = np.array(self.n_time*list(self.A1))
            AA2 = np.array(self.n_time*list(self.A2))
            chunks = np.array(self.n_time*[self.n_bl], dtype=np.int)
            auvw = mbu.antenna_uvw(self.UVW, AA1, AA2, chunks,
                                   nr_of_antenna=self.n_ant)

        return auvw[lt:ut, la:ua, l:u]

######### Sink Provider ########################################################

class RFISinkProvider(SinkProvider):
    """
    Receives data from montblanc via data sink methods,
    which have the following signature

    .. code-block:: python

        def model_vis(self, context):
            print context. data
    """
    def __init__(self, vis, rfi_run, time_step, noise):
        self.vis = vis
        self.rfi_run = rfi_run
        self.time_step = time_step
        self.noise = noise

    def name(self):
        """ Name of the Sink Provider """
        return self.__class__.__name__

    def model_vis(self, context):
        """ Receive model visibilities from Montblanc in `context.data` """
        extents = context.array_extents(context.name)
        (lt, ut), (lbl, ubl), (lc, uc), (lp, up) = extents
        context_shape = context.data.shape

        # noise = self.noise*context.data
        complex_noise = np.random.randn(*context_shape) * self.noise + \
                        np.random.randn(*context_shape) * self.noise * 1j
        if self.rfi_run:
            i = self.time_step
            self.vis[i, lbl:ubl, lc:uc, lp:up] = context.data + complex_noise
        else:
            self.vis[lt:ut, lbl:ubl, lc:uc, lp:up] = context.data + complex_noise
