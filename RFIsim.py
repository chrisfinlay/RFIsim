import numpy as np

import montblanc
from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sources import FitsBeamSourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider
import montblanc.util as mbu

import time as tme

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Internal imports
from utils.uv_sim.uvgen import UVCreate
from utils.sat_sim.sim_sat_paths import get_lm_tracks, radec_to_lm
from utils.sat_sim.sim_sat_spectra import get_sat_spectra
from utils.bandpass.bandpass_gains import get_bandpass_and_gains
from utils.catalogues.get_ast_sources import inview
from utils.write_to_h5 import save_output, save_input

######## Source Provider #######################################################

class RFISourceProvider(SourceProvider):
    """
    Supplies data to montblanc via data source methods,
    which have the following signature.

    .. code-block:: python

        def point_lm(self, context)
            ...
    """
    def name(self):
        """ Name of Source Provider """
        return self.__class__.__name__

    def updated_dimensions(self):
        """ Inform montblanc about dimension sizes """
        if j==0:
            return [("ntime", ntime),              # Timesteps
                    ("nchan", nchan),              # Channels
                    ("na", n_ant),                    # Antenna
                    ("nbl", n_ant*(n_ant-1)/2),          # Baselines
                    ("npsrc", len(lm_coords)),     # Number of point sources
                    ("ngsrc", 0)]                  # Number of gaussian sources
                    # ("ngsrc", len(gauss_sources))] # Number of gaussian sources
        else:
            return [("ntime", time_steps),         # Timesteps
                    ("nchan", nchan),              # Channels
                    ("na", n_ant),                    # Antenna
                    ("nbl", n_ant*(n_ant-1)/2),          # Baselines
                    ("npsrc", 0),                  # Number of point sources
                    ("ngsrc", len(gauss_sources))] # Number of gaussian sources

    def point_lm(self, context):
        """ Supply point source lm coordinates to montblanc """

        # Shape (npsrc, 2)
        (ls, us), _ = context.array_extents(context.name)

        return np.asarray(lm_coords, dtype=context.dtype)[ls:us, :]

    def point_stokes(self, context):
        extents = context.dim_extents('npsrc', 'ntime', 'nchan')
        (lp, up), (lt, ut), (lc, uc) = extents
        # (npsrc, ntime, nchan, 4)

        spec = np.ones((1, time_steps, 1, 1))
        spec = spectra*spec
        return spec[lp:up, lt:ut, lc:uc, :]

######## New ############################################################################

    def gaussian_lm(self, context):
        """ Returns an lm coordinate array to Montblanc. """

        # lm = np.empty(context.shape, context.dtype)
        lm = np.empty((len(gauss_sources), 2), context.dtype)

        # Get the extents of the time, baseline and chan dimension
        (lg, ug) = context.dim_extents('ngsrc')

        ra = gauss_sources['RA']
        dec = gauss_sources['DEC']
        phase_centre = [target_ra, target_dec]

        lm[:,0], lm[:,1] = radec_to_lm(ra, dec, phase_centre)

        return lm[lg:ug]

    def gaussian_shape(self, context):
        """ Returns a Gaussian shape array to Montblanc """

        (lg, ug) = context.dim_extents('ngsrc')

        emaj = np.deg2rad(gauss_sources['MajAxis'].values)
        emin = np.deg2rad(gauss_sources['MinAxis'].values)
        pa = np.deg2rad(gauss_sources['PA'].values)

        # gauss = np.empty(context.shape, dtype=context.dtype)
        gauss = np.empty((3, len(gauss_sources)), context.dtype)

        gauss[0,:] = emaj * np.sin(pa)
        gauss[1,:] = emaj * np.cos(pa)
        emaj[emaj == 0.0] = 1.0
        gauss[2,:] = emin / emaj

        return gauss[:,lg:ug]

    def gaussian_stokes(self, context):
        """ Return a stokes parameter array to Montblanc """

        # Get the extents of the time, baseline and chan dimension
        extents = context.dim_extents('ngsrc', 'ntime', 'nchan')
        (lg, ug), (lt, ut), (lc, uc) = extents
        print(context.shape)
        # stokes = np.empty(context.shape, context.dtype)
        stokes = np.empty((len(gauss_sources), 1, uc-lc, 4), context.dtype)

        stokes[:,:,:,0] = gauss_sources['Flux'].values[:,None,None]
        stokes[:,:,:,1:] = 0.0

        return stokes[lg:ug]

######################################################################################

    # def direction_independent_effects(self, context):
    #     # (ntime, na, nchan, npol)
    #     extents = context.dim_extents('ntime', 'na', 'nchan', 'npol')
    #     (lt, ut), (la, ua), (lc, uc), (lp, up) = extents
    #
    #     bp = np.ones((time_steps, 1, 1, 1))
    #     bp = bandpass*bp
    #     return bp[lt:ut, la:ua, lc:uc, lp:up]

    def uvw(self, context):
        """ Supply UVW antenna coordinates to montblanc """

        # Shape (ntime, na, 3)
        (lt, ut), (la, ua), (l, u) = context.array_extents(context.name)

        if j==0:
            idx = np.arange(i*2016, (i+1)*2016)
            auvw = mbu.antenna_uvw(UVW[idx], A1, A2, np.array([2016,]),
                                   nr_of_antenna=n_ant)
        else:
            AA1 = []
            AA2 = []
            for _ in range(UVW.shape[0]/2016):
                AA1 += list(A1)
                AA2 += list(A2)
            AA1 = np.array(AA1)
            AA2 = np.array(AA2)
            chunks = np.array(UVW.shape[0]/2016*[n_ant*(n_ant-1)/2], dtype=np.int)
            auvw = mbu.antenna_uvw(UVW, AA1, AA2, chunks, nr_of_antenna=n_ant)

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
    def name(self):
        """ Name of the Sink Provider """
        return self.__class__.__name__

    def model_vis(self, context):
        """ Receive model visibilities from Montblanc in `context.data` """
        extents = context.array_extents(context.name)
        (lt, ut), (lbl, ubl), (lc, uc), (lp, up) = extents
        global vis
        # noise = context.data/20
        context_shape = (ut-lt, ubl-lbl, uc-lc, up-lp)
        # complex_noise = np.random.randn(*context_shape) * noise + \
                        # np.random.randn(*context_shape) * noise * 1j
        if j==0:
            vis[i, lbl:ubl, lc:uc, lp:up] = context.data# + complex_noise
        else:
            vis[lt:ut, lbl:ubl, lc:uc, lp:up] = context.data# + complex_noise

        return vis

########## Configuration #######################################################

# Configure montblanc solver with a memory budget of 1.7GB
# and set it to double precision floating point accuracy

slvr_cfg = montblanc.rime_solver_cfg(mem_budget=int(1.7*1024*1024*1024),
                                     dtype='double')

######### Solver Definition ####################################################

def call_solver():
    # Create montblanc solver
    with montblanc.rime_solver(slvr_cfg) as slvr:

        FITSfiles = 'utils/beam_sim/beams/FAKE_$(corr)_$(reim).fits'

        # Create RFI Source and Sink Providers
        source_provs = [RFISourceProvider(),
                        # FitsBeamSourceProvider(FITSfiles)
                       ]
        sink_provs = [RFISinkProvider()]

        # Call solver, supplying source and sink providers
        slvr.solve(source_providers=source_provs,
                sink_providers=sink_provs)


########## Run Simulation ######################################################
start = tme.time()
# Set number of channels and antennas

ntime = 1
nchan = 4096
n_ant = 64

# Define target, track length and integration time

target_ra = 21.4439
target_dec = -30.713199999999997
phase_centre = [target_ra, target_dec]
# target_ra, target_dec = 0., 0.
tracking_hours = 48./3600
integration_secs = 8
obs_date = '2018/11/07'

# Create UV tracks

direction = 'J2000,'+str(target_ra)+'deg,'+str(target_dec)+'deg'
uv = UVCreate(antennas='utils/uv_sim/MeerKAT.enu.txt', direction=direction,
              tel='meerkat', coord_sys='enu')
ha = -tracking_hours/2, tracking_hours/2
transit, UVW = uv.itrf2uvw(h0=ha, dtime=integration_secs/3600., date=obs_date)

#### Get antenna baseline pairings #############################################

A1, A2 = np.triu_indices(n_ant, 1)

##### Get astronomical sources #################################################

gauss_sources = inview(phase_centre, radius=10, min_flux=0.5)

#### Get lm tracks of satellites ##### lm shape (time_steps, vis_sats+1, 2) ####
lm = get_lm_tracks(phase_centre, transit, tracking_hours,
                   integration_secs)
time_steps, n_sats = lm.shape[:2]

###### Get satellite spectra ###################################################

spectra = get_sat_spectra(n_chan=nchan, n_sats=n_sats, n_time=time_steps)

###### Get bandpass ############################################################

bandpass, auto_gains, cross_gains = get_bandpass_and_gains()

###### Save input data #########################################################

save_file = 'ra=' + str(target_ra) + '_dec=' + str(target_dec) + \
            '_int_secs=' + str(integration_secs) + \
            '_track-time=' + str(round(tracking_hours, 1)) + \
            'hrs_nants=' + str(n_ant) + '_nchan=' + str(nchan) + '.h5'

save_input(save_file, phase_centre, lm, UVW, A1, A2, spectra, bandpass,
           auto_gains, cross_gains)

# Run simulation twice - once with RFI and once without
run_start = []

for j in range(2):

    run_start.append(tme.time())
    print('\n\nInitialization time : {} s\n\n'.format(run_start[j]-start))

    vis = np.zeros(shape=(len(lm), n_ant*(n_ant-1)/2, nchan, 4),
                          dtype=np.complex128)

    if j==1:
        call_solver()
    else:
        for i in range(len(lm)):
            lm_coords = lm[i]
            call_solver()

    # Save output
    save_output(save_file, vis, clean=j)

    print('\n\nCompletion time : {} s\n\n'.format(tme.time()-start))
