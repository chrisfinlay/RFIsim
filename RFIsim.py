# System imports
import argparse
import numpy as np
import time as tme
import datetime
import os

# Internal imports
from utils.helper.write_to_h5 import save_output, save_input
from utils.telescope.uv_sim.uvgen import UVCreate
from utils.telescope.bandpass.bandpass_gains import get_bandpass_and_gains
from utils.rfi.sat_sim.sim_sat_paths import get_lm_tracks, radec_to_lm
from utils.rfi.rfi_spectra.sim_rfi_spectra import get_rfi_spectra
from utils.rfi.horizon_sim.horizon_sources import get_horizon_lm_tracks
from utils.astronomical.get_ast_sources import inview, find_closest

######### Arg Parser ###########################################################
def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        help='Config file')
    return parser

args = create_parser().parse_args()

######## Read Config ###########################################################
config = load_config(args.config)



########## Run Simulation ######################################################
start = tme.time()

######## Create UV tracks ######################################################

target_ra, target_dec, target_flux = find_closest(target_ra, target_dec, min_flux)
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

astro_srcs = inview(phase_centre, sky_radius, min_flux)

#### Get lm tracks of satellites ##### lm shape (time_steps, vis_sats+1, 2) ####
sat_lm, obs_times = get_lm_tracks(phase_centre, transit, tracking_hours,
                                  integration_secs)

sat_lm = sat_lm[:,:n_sats,:]


###### Get horizon rfi sources #################################################
horizon_lm = get_horizon_lm_tracks(phase_centre, transit, tracking_hours,
                          integration_secs)

##### Join RFI source paths ####################################################
rfi_lm = np.concatenate((sat_lm, horizon_lm), axis=1)
n_rfi = rfi_lm.shape[1]

###### Get satellite spectra ###################################################

rfi_spectra = get_rfi_spectra(n_chan=n_chan, n_rfi=n_rfi,
                              n_time=time_steps, type=args.rfi_sig)

###### Get bandpass ############################################################

bandpass, auto_gains, cross_gains = get_bandpass_and_gains(target_flux,
                                                           obs_times)

###### Save input data #########################################################

date = datetime.datetime.strptime(transit,
                                  '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')

save_file = 'date={0}_ra={1}_dec={2}_int_secs={3}_timesteps={4}_nants={5}' \
            '_noise={6}_rfi={7}.h5'.format(date, round(target_ra, 2),
                                           round(target_dec, 2),
                                           integration_secs, time_steps,
                                           n_ant, round(noise,3),
                                           args.rfi_sig)

save_file = os.path.join(save_dir, save_file)
save_input(save_file, phase_centre, astro_srcs, rfi_lm, UVW, A1, A2,
           rfi_spectra, bandpass, freqs, auto_gains, cross_gains, obs_times)

with open(args.timing, 'a') as t:
    now = str(datetime.datetime.now()+datetime.timedelta(hours=2))
    t.write('\n-----------------------'+now+'-------------------------\n\n')
    t.write(save_file)
    t.write('\n\nInitialization time   : {} s\n'.format(tme.time()-start))

### Run simulation twice - once with RFI and once without ######################
run_start = []

for j in range(2):

    run_start.append(tme.time())

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
    if j==0:
        dirty_time = round(tme.time()-run_start[0], 2)
        with open(args.timing, 'a') as t:
            t.write('\nDirty Completion time : {} s\n'.format(dirty_time))

clean_time = round(tme.time()-run_start[1], 2)
total_time = round(tme.time()-start, 2)
print('\n\nDirty Completion time : {} s\n'.format(dirty_time))
print('\nClean Completion time : {} s\n'.format(clean_time))
print('\nTotal Completion time : {} s\n\n'.format(total_time))

with open(args.timing, 'a') as t:
    t.write('\nClean Completion time : {} s\n'.format(clean_time))
    t.write('\nTotal Completion time : {} s\n\n'.format(total_time))
