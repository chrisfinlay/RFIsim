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
from utils.RFIProviders import RFISourceProvider, RFISinkProvider
from utils.helper.write_to_h5 import save_output, save_input
from utils.telescope.uv_sim.uvgen import UVCreate
from utils.telescope.bandpass.bandpass_gains import get_bandpass_and_gains
from utils.rfi.sat_sim.sim_sat_paths import get_lm_tracks, radec_to_lm
from utils.rfi.rfi_spectra.sim_rfi_spectra import get_rfi_spectra
from utils.rfi.horizon_sim.horizon_sources import get_horizon_lm_tracks
from utils.astronomical.get_ast_sources import inview, find_closest

########## Configuration #######################################################

# Configure montblanc solver with a memory budget of 1.7GB
# and set it to double precision floating point accuracy

slvr_cfg = montblanc.rime_solver_cfg(mem_budget=int(1.7*1024*1024*1024),
                                     dtype='double')

######### Solver Definition ####################################################

def call_solver(rfi_run, time_step):
    # Create montblanc solver
    with montblanc.rime_solver(slvr_cfg) as slvr:

        FITSfiles = 'utils/telescope/beam_sim/beams/FAKE_$(corr)_$(reim).fits'

        source = RFISourceProvider(rfi_run, n_time, n_chan, n_ant,
                                  bandpass, gauss_sources, rfi_spectra,
                                  rfi_lm, phase_centre, time_step, A1, A2, UVW)
        # Create RFI Source and Sink Providers
        source_provs = [source,
                        FitsBeamSourceProvider(FITSfiles)
                       ]
        sink_provs = [RFISinkProvider(vis, rfi_run, time_step)]

        # Call solver, supplying source and sink providers
        slvr.solve(source_providers=source_provs,
                sink_providers=sink_provs)


########## Run Simulation ######################################################
start = tme.time()
########## Set number of channels and antennas #################################

n_time = 1
n_chan = 4096
n_ant = 64

min_flux = 0.5      # Jy
sky_radius = 10     # degrees

######## Define target, track length and integration time ######################

t_steps = 3
tracking_hours = t_steps*8./3600
integration_secs = 8
obs_date = '2018/11/07'
target_ra = 21.4439
target_dec = -30.713199999999997

######## Create UV tracks ######################################################

target_ra, target_dec = find_closest(target_ra, target_dec, min_flux)
phase_centre = [target_ra, target_dec]
direction = 'J2000,'+str(target_ra)+'deg,'+str(target_dec)+'deg'
uv = UVCreate(antennas='utils/telescope/uv_sim/MeerKAT.enu.txt',
              direction=direction, tel='meerkat', coord_sys='enu')
ha = -tracking_hours/2, tracking_hours/2
transit, UVW = uv.itrf2uvw(h0=ha, dtime=integration_secs/3600., date=obs_date)

#### Get antenna baseline pairings #############################################

A1, A2 = np.triu_indices(n_ant, 1)

##### Get astronomical sources #################################################

gauss_sources = inview(phase_centre, sky_radius, min_flux)

#### Get lm tracks of satellites ##### lm shape (time_steps, vis_sats+1, 2) ####
sat_lm = get_lm_tracks(phase_centre, transit, tracking_hours,
                   integration_secs)

###### Get horizon rfi sources #################################################
horizon_lm = get_horizon_lm_tracks(phase_centre, transit, tracking_hours,
                          integration_secs)
                          
##### Join RFI source paths ####################################################
rfi_lm = np.concatenate((sat_lm, horizon_lm), axis=1)
time_steps = rfi_lm.shape[0]
n_rfi = rfi_lm.shape[1]-1

###### Get satellite spectra ###################################################

rfi_spectra = get_rfi_spectra(n_chan=n_chan, n_rfi=n_rfi, n_time=time_steps)

###### Get bandpass ############################################################

bandpass, auto_gains, cross_gains = get_bandpass_and_gains()

###### Save input data #########################################################

save_file = 'ra=' + str(target_ra) + '_dec=' + str(target_dec) + \
            '_int_secs=' + str(integration_secs) + \
            '_track-time=' + str(round(tracking_hours, 1)) + \
            'hrs_nants=' + str(n_ant) + '_nchan=' + str(n_chan) + '.h5'

save_input(save_file, phase_centre, rfi_lm, UVW, A1, A2, rfi_spectra, bandpass,
           auto_gains, cross_gains)

### Run simulation twice - once with RFI and once without ######################
run_start = []

for j in range(2):

    run_start.append(tme.time())
    print('\n\nInitialization time : {} s\n\n'.format(run_start[j]-start))

    vis = np.zeros(shape=(len(rfi_lm), n_ant*(n_ant-1)/2, n_chan, 4),
                          dtype=np.complex128)

    if j==1:
        n_time = time_steps
        call_solver(rfi_run=False, time_step=0)
    else:
        for i in range(len(rfi_lm)):
            call_solver(rfi_run=True, time_step=i)

    # Save output
    save_output(save_file, vis, clean=j)

dirty_time = round(run_start[1]-run_start[0], 2)
clean_time = round(tme.time()-run_start[1], 2)
total_time = round(tme.time()-start, 2)
print('\n\nDirty Completion time : {} s\n'.format(dirty_time))
print('\nClean Completion time : {} s\n'.format(clean_time))
print('\nTotal Completion time : {} s\n\n'.format(total_time))
