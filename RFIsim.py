# System imports
import argparse
import numpy as np
import time as tme
import datetime
import os

# Montblanc imports
import montblanc
from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sources import FitsBeamSourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider
import montblanc.util as mbu

# Internal imports
from utils.RFIProviders import RFISourceProvider, RFISinkProvider
from utils.helper.write_to_h5 import save_output, save_input
from utils.telescope.uv_sim.uvgen import UVCreate
from utils.telescope.bandpass.bandpass_gains import get_bandpass_and_gains
from utils.rfi.sat_sim.sim_sat_paths import get_lm_tracks, radec_to_lm
from utils.rfi.rfi_spectra.sim_rfi_spectra import get_rfi_spectra
from utils.rfi.horizon_sim.horizon_sources import get_horizon_lm_tracks
from utils.astronomical.get_ast_sources import inview, find_closest

######### Arg Parser ###########################################################
def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y/%m/%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntime", default=5, type=int,
                        help="Number of timesteps")
    parser.add_argument("--intsecs", default=8, type=int,
                        help="Integration time per time step in seconds"),
    parser.add_argument("--nant", default=16, type=int,
                        help="Number of antenna")
    parser.add_argument("--ra", default=0.0, type=float,
                        help="""Right ascension of target direction in
                        decimal degrees""")
    parser.add_argument("--dec", default=0.0, type=float,
                        help="""Declination of target direction in
                        decimal degrees""")
    parser.add_argument("--date", default='2018/11/07', type=valid_date,
                        help="Date of the observation. Format YYYY/MM/DD")
    parser.add_argument("--minflux", default=0.5, type=float,
                        help="Minimum flux of astronomical sources in Jy")
    parser.add_argument("--radius", default=10, type=float,
                        help="""Radius around target in which to include
                        astronomical sources in degrees""")
    parser.add_argument("--gpu", default=0, type=int,
                        help="""GPU id e.g 0. If you want to run on the CPU
                        use -1.""")


    return parser

args = create_parser().parse_args()

if args.gpu == -1:
    os.environ["CUDA_VISIBLE_DEVICES"]=""
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
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

time_steps = args.ntime
integration_secs = args.intsecs
n_ant = args.nant
target_ra = args.ra
target_dec = args.dec
obs_date = args.date
min_flux = args.minflux
sky_radius = args.radius

########## Fixed Parameters ####################################################

n_chan = 4096
n_bl = n_ant*(n_ant-1)/2

######## Define track length ###################################################

tracking_hours = float(time_steps*integration_secs)/3600

######## Create UV tracks ######################################################

target_ra, target_dec = find_closest(target_ra, target_dec, min_flux)
phase_centre = [target_ra, target_dec]
print("""\nMoving phase centre to closest astronomical target @ \
         RA:{} , DEC:{}\n""".format(*np.round(phase_centre, 2)))

direction = 'J2000,'+str(target_ra)+'deg,'+str(target_dec)+'deg'
uv = UVCreate(antennas='utils/telescope/uv_sim/MeerKAT.enu.txt',
              direction=direction, tel='meerkat', coord_sys='enu', n_ant=n_ant)
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
n_rfi = rfi_lm.shape[1]-1

###### Get satellite spectra ###################################################

rfi_spectra = get_rfi_spectra(n_chan=n_chan, n_rfi=n_rfi, n_time=time_steps)

###### Get bandpass ############################################################

bandpass, auto_gains, cross_gains = get_bandpass_and_gains()

###### Save input data #########################################################

save_file = 'date=' + str(obs_date) + '_ra=' + str(round(target_ra, 2)) + \
            '_dec=' + str(round(target_dec, 2)) + '_int_secs=' + \
            str(integration_secs) + '_track-time=' + \
            str(round(tracking_hours, 1)) + 'hrs_nants=' + str(n_ant) + \
            '_nchan=' + str(n_chan) + '.h5'

save_input(save_file, phase_centre, rfi_lm, UVW, A1, A2, rfi_spectra, bandpass,
           auto_gains, cross_gains)

### Run simulation twice - once with RFI and once without ######################
run_start = []

for j in range(2):

    run_start.append(tme.time())
    print('\n\nInitialization time : {} s\n\n'.format(run_start[j]-start))

    vis = np.zeros(shape=(len(rfi_lm), n_bl, n_chan, 4), dtype=np.complex128)

    if j==1:
        n_time = time_steps
        call_solver(rfi_run=False, time_step=0)
    else:
        n_time = 1
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
