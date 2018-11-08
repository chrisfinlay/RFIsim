import h5py
import numpy as np

import montblanc
from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sources import FitsBeamSourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider
import montblanc.util as mbu

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Internal imports
from utils.uv_sim.uvgen import UVCreate
from utils.sat_sim.sim_sat_paths import get_lm_tracks

######## Source Provider #######################################################

class CustomSourceProvider(SourceProvider):
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
        return [("ntime", ntime),          # Timesteps
                ("nchan", nchan),          # Channels
                ("na", na),                # Antenna
                ("nbl", na*(na-1)/2),      # Baselines
                ("npsrc", len(lm_coords))] # Number of point sources

    def point_lm(self, context):
        """ Supply point source lm coordinates to montblanc """

        # Shape (npsrc, 2)
        (ls, us), _ = context.array_extents(context.name)

        return np.asarray(lm_coords[ls:us], dtype=context.dtype)[ls:us, :]

    def point_stokes(self, context):
        extents = context.dim_extents('npsrc', 'ntime', 'nchan')
        (lp, up), (lt, ut), (lc, uc) = extents
        # (npsrc, ntime, nchan, 4)

        return spectra[lp:up, lt:up, lc:uc, :]

    def direction_independent_effects(self, context):
        # (ntime, na, nchan, npol)
        extents = context.dim_extents('ntime', 'na', 'nchan', 'npol')
        (lt, ut), (la, ua), (lc, uc), (lp, up) = extents
        return bandpass[lt:ut, la:ua, lc:uc, lp:up]

#     def observed_vis

#     def model_vis(self, context):
#         if self.noise != 0:
#             n = np.random.randn(*context.shape) * self.noise + \
#                 np.random.randn(*context.shape) * self.noise * 1j
#         else:
#             n = np.zeros(shape=context.shape, dtype=context.dtype)
#         return n

    def uvw(self, context):
        """ Supply UVW antenna coordinates to montblanc """

        # Shape (ntime, na, 3)
        (lt, ut), (la, ua), (l, u) = context.array_extents(context.name)

        idx = np.arange(i*2016, (i+1)*2016)
        auvw = mbu.antenna_uvw(UVW[idx], A1, A2, np.array([2016,]),
                               nr_of_antenna=na)

        return auvw[lt:ut, la:ua, l:u]

######### Sink Provider ########################################################

class CustomSinkProvider(SinkProvider):
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
        vis[i] = context.data
#         try:
#             vis = np.load("vis.npy")
#             vis = np.concatenate((vis, context.data), axis=0)
#             np.save("vis.npy", vis)
#         except:
#             np.save("vis.npy", context.data)


########## Configuration #######################################################

# Configure montblanc solver with a memory budget of 6GB
# and set it to double precision floating point accuracy

slvr_cfg = montblanc.rime_solver_cfg(mem_budget=6*1024*1024*1024,
                                     dtype='double')

########## Run Simulation ######################################################

# Set number of channels and antennas

ntime = 1
nchan = 4096
na = 64

# Define target, track length and integration time

target_ra = 21.4439
target_dec = -30.713199999999997
# target_ra, target_dec = 0., 0.
tracking_hours = 0.1
integration_secs = 8
obs_date = '2018/11/07'

# Create UV tracks

direction = 'J2000,'+str(target_ra)+'deg,'+str(target_dec)+'deg'
uv = UVCreate(antennas='utils/uv_sim/MeerKAT.enu.txt', direction=direction,
              tel='meerkat', coord_sys='enu')
ha = -tracking_hours/2, tracking_hours/2
transit, UVW = uv.itrf2uvw(h0=ha, dtime=integration_secs/3600., date=obs_date)

# Get antenna baseline pairings

nant = 64
A1 = np.empty(nant*(nant-1)/2, dtype=np.int32)
A2 = np.empty(nant*(nant-1)/2, dtype=np.int32)
k = 0
for i in range(nant):
    for j in range(i+1,nant):
        A1[k] = i
        A2[k] = j
        k += 1

#### Get lm tracks of satellites ##### lm shape (time_steps, vis_sats, 2) ######
lm = get_lm_tracks(target_ra, target_dec, transit, tracking_hours,
                   integration_secs)
time_steps, n_sats = lm.shape[:2]

astro_lm = np.array([
                    [0.0, 0.0],
                    # [0.1, 0.0]
                    ])[None,:,:]*np.ones((time_steps, 1, 1))

all_lm = np.concatenate((astro_lm, lm), axis=1)

# Create frequency spectrum array
np.random.seed(123)

spectra = np.load('utils/sat_sim/sat_spectra/Satellite_Frequency_Spectra.npy')
perm = np.random.permutation(len(spectra))
spectra[perm] = spectra

n_spectra, channels = spectra.shape
n_srcs = 1
wraps = n_sats/n_spectra+1
wrapped_spectra = np.array([spectra for i in range(wraps)])
wrapped_spectra = wrapped_spectra.reshape(-1, channels)
sat_spectrum = wrapped_spectra[:n_sats]
sat_spectrum *= (1e5*np.random.rand(n_sats)+1e4)[:,None]

full_spectrogram = np.zeros((n_sats+n_srcs, 1, nchan, 1))
freq_i = np.random.randint(0, nchan-channels, size=n_sats)
freq_f = freq_i + channels
for i in range(n_sats):
    full_spectrogram[n_srcs+i, 0, freq_i[i]:freq_f[i], 0] = sat_spectrum[i]
#### To be changed to accomodate arbitrary astronomical sources ################

freqs = np.linspace(800, 1800, 4096)
source_spec = (freqs[-1]/freqs)**0.667
full_spectrogram[0,:,:,0] = 2*np.ones((1, 1))*source_spec[None,:]

################################################################################

bandpass_file = 'utils/bandpass/MeerKAT_Bandpass_HH-HV-VH-VV.npy'
bandpass = np.load(bandpass_file).astype(np.complex128)
FITSfiles = 'utils/beam_sim/beams/FAKE_$(corr)_$(reim).fits'

# Set file name to save data to

save_file = 'ra=' + str(target_ra) + '_dec=' + str(target_dec) + \
            '_int_secs=' + str(integration_secs) + \
            '_track-time=' + str(round(tracking_hours, 1)) + \
            'hrs_nants=' + str(nant) + '_nchan=' + str(nchan) + '.h5'


# Need to change the number of sources variably
with h5py.File(save_file, 'a') as fp:
    fp['/input/lm'] = all_lm
    fp['/input/UVW'] = UVW
    fp['/input/A1'] = A1
    fp['/input/A2'] = A2
    fp['/input/bandpass'] = bandpass

# Stokes parameters (I, Q, U, V)
stokes_sats = np.random.randint(1, 4, size=n_sats)
signs = (-1)**np.random.randint(0, 2, size=n_sats)
lm_stokes_sats = np.zeros((n_sats, 4))
lm_stokes_sats[:, 0] = 1.0
for i in range(n_sats):
    lm_stokes_sats[i, stokes_sats[i]] = signs[i]

stokes_srcs = np.array([
                        [1.0, 0.0, 0.0, 0.0],
                        # [1.0, 0.0, 0.0, 0.0]
                        ])

lm_stokes = np.concatenate((stokes_srcs, lm_stokes_sats), axis=0)

# Save input spectra
s = np.asarray(lm_stokes, dtype=np.float64)[:,None,None,:]
spectra = s*full_spectrogram

with h5py.File(save_file, 'a') as fp:
    fp['/input/spectra'] = spectra

# Save input bandpasses
np.random.seed(123)
antenna_gains_auto = 200*np.random.rand(na) + 50
antenna_gains_cross = 70*np.random.rand(na) + 10

bandpass[:,:,:,[0,3]] *= antenna_gains_auto[None, :, None, None]
bandpass[:,:,:,[1,2]] *= antenna_gains_cross[None, :, None, None]

with h5py.File(save_file, 'a') as fp:
    fp['/input/auto_pol_gains'] = antenna_gains_auto
    fp['/input/cross_pol_gains'] = antenna_gains_cross

# Run simulation twice - once with RFI and once without

for j in range(2):

    vis = np.zeros(shape=(len(lm), na*(na-1)/2, nchan, 4), dtype=np.complex128)

    if j==1:
        spectra = (spectra[0])[None, :, :, :]
        lm_stokes = [lm_stokes[0]]


    for i in range(len(lm)):
        # LM coordinates
        if j==0:
            lm_coords = all_lm[i]
        else:
            lm_coords = all_lm[0, :n_srcs, :]

        # Create montblanc solver
        with montblanc.rime_solver(slvr_cfg) as slvr:

#         ms_mgr = MeasurementSetManager('MeerKAT_data/AP_Lib_0.ms', slvr_cfg)

            # Create Customer Source and Sink Providers
            source_provs = [CustomSourceProvider(),
                            FitsBeamSourceProvider(FITSfiles)
                           ]
            sink_provs = [CustomSinkProvider()]

            # Call solver, supplying source and sink providers
            slvr.solve(source_providers=source_provs,
                    sink_providers=sink_provs)

    # Save output
    if j==0:
        with h5py.File(save_file, 'a') as fp:
            fp['/output/vis_dirty'] = vis

    else:
        with h5py.File(save_file, 'a') as fp:
            fp['/output/vis_clean'] = vis
