import numpy as np

def get_corr_idx(tele, a1, a2, pol1, pol2, pointing):
    """
    Get the correlation product index for a specific correlation and pointing.

    Parameters:
    -----------
    tele: hirax_transfer.core.HIRAXSurvey
        HIRAX Telescope object.
    a1: int
        Antenna 1 index.
    a2: int
        Antenna 2 index.
    pol1: char
        Feed polarization. H or V.
    pol2: char
        Feed polarization. H or V.
    pointing: float
        The N-S pointing from zenith of the observation.

    Returns:
    --------
    corr_idx: int
        The index of the correlation product and pointing.
    """

    n_pnt = len(tele.pointings)
    n_ants = tele.nfeed/2/n_pnt

    if a1>=n_ants or a2>=n_ants:
        print('The are only {0} antennas. Arguments a1 and a2 should be an integer in 0<=x<{0}.\n'.format(n_ants))
        return

    if pol1.upper() not in ['H', 'V']:
        print('H or V needed in pol1 argument.\n')
        return
    if pol2.upper() not in ['H', 'V']:
        print('H or V needed in pol2 argument.\n')
        return

    p_idx = np.where(pointing==tele.pointings)[0]
    if len(p_idx)==0:
        print('Could not find the specified pointing. \nAvailable pointings are {}.\n'.format(list(tele.pointings)))
        return

    pol1 = 0 if pol1.upper()=='H' else 1
    pol2 = 0 if pol2.upper()=='H' else 1

    feed_idx_1 = np.where((tele.beamclass==pol1))[0][p_idx*n_ants+a1]
    feed_idx_2 = np.where((tele.beamclass==pol2))[0][p_idx*n_ants+a2]

    return int(tele.feedmap[feed_idx_1, feed_idx_2])

def avg_redundant_and_reshape(tele, vis):
    """
    Average redundant visibilities and reshape for HIRAX data product.

    Parameters:
    -----------
    tele: hirax_transfer.core.HIRAXSurvey
        HIRAX Telescope object.
    vis: np.array (n_point,2,2,n_bl,n_freqs)
        The visibilty array.

    Returns:
    --------
    new_vis: np.array (n_corrs*n_point,n_freqs)
        The new visibilty array with redundant baselines averaged.
    """

    n_pnt = len(tele.pointings)
    n_ants = tele.nfeed/2/n_pnt

    new_vis = np.zeros((tele.nbase, tele.nfreq), np.complex128)
    red_cnt = np.zeros((tele.nbase, tele.nfreq))

    a1, a2 = np.triu_indices(n_ants, 1)
    for j, point in enumerate(tele.pointings):
        corr_idx_HH = np.array([get_corr_idx(tele=tele, a1=a1[i], a2=a2[i],
                                    pol1='h', pol2='h', pointing=point) for i in range(len(a1))])
        corr_idx_HV = np.array([get_corr_idx(tele=tele, a1=a1[i], a2=a2[i],
                                    pol1='h', pol2='v', pointing=point) for i in range(len(a1))])
        corr_idx_VH = np.array([get_corr_idx(tele=tele, a1=a1[i], a2=a2[i],
                                    pol1='v', pol2='h', pointing=point) for i in range(len(a1))])
        corr_idx_VV = np.array([get_corr_idx(tele=tele, a1=a1[i], a2=a2[i],
                                    pol1='v', pol2='v', pointing=point) for i in range(len(a1))])

        new_vis[corr_idx_HH,:] += vis[0,0,:,:]
        new_vis[corr_idx_HV,:] += vis[0,1,:,:]
        new_vis[corr_idx_VH,:] += vis[1,0,:,:]
        new_vis[corr_idx_VV,:] += vis[1,1,:,:]

        red_cnt[corr_idx_HH,:] += 1
        red_cnt[corr_idx_HV,:] += 1
        red_cnt[corr_idx_VH,:] += 1
        red_cnt[corr_idx_VV,:] += 1

    return np.divide(new_vis, red_cnt, out=np.zeros_like(new_vis), where=red_cnt!=0)

def add_rfi_to_H5(path_to_timestream, red_V):
    """
    Add the visibility contribution from RFI into the visibility H5 files.

    Parameters:
    -----------
    path_to_timestream: str
        Path to the directory containing each frequency channel.
    red_V: np.array (t_steps,n_bl,n_freqs)
        The visibilities on each unique correlation product.
    """

    for i in range(red_V.shape[-1]):
        path = os.path.join(path_to_timestream, str(i))
        with h5py.File(os.path.join(path, 'timestream.hdf5'), 'r+') as fp:
            timestream = fp[('timestream')]
            timestream[...] += red_V[:,3:,i].T

def get_rfi_vis_all_time(tele, date, n_time):
    """
    Calculate the visibilities for all time steps.

    Parameters:
    ----------
    tele: hirax_transfer.core.HIRAXSurvey
        HIRAX Telescope object.
    date: str
        The date on which to simulate RFI.
    n_time: int
        The number of time steps to simulate.

    Return:
    -------
    red_V: np.array (t_steps,n_bl,n_freqs)
        The visibilities
    """

    with open('prod_params.yaml') as f:
        tele = HIRAXSurvey.from_config(yaml.load(f, Loader=yaml.BaseLoader)['telescope'])

    n_ants = tele.nfeed/2/len(tele.pointings)

    enu = tele.feedpositions[:n_ants,:]
    enu = np.concatenate([enu, 1110.*np.ones(len(enu))[:,None]], axis=1)

    freqs = tele.frequencies

    gps_ants = enu_to_gps_el([-30.69, 21.57527778], enu)

    midnight = datetime.datetime(2019, 6, 28, 0, 0, 0)
    one_day = datetime.timedelta(seconds=3600*24)
    time_step = datetime.timedelta(seconds=3600*24/n_time)

    tles = tle_load.tle(get_archival_tles(midnight, midnight+one_day))

    red_V = np.zeros((n_time,tele.nbase,len(freqs)), np.complex128)

    for i in range(n_time):
        el, distances, seps = np.array([get_dist_and_sep(tles[tles.keys()[0]], gps,
                                                            0., midnight+time_step*i) for gps in gps_ants]).T
        delays = get_time_delays(distances)
        RFI_intensity = 1e11
        rfi_I = RFI_intensity*np.ones((1, len(freqs)))
        rfi_Q = RFI_intensity*np.zeros((1, len(freqs)))
        rfi_U = np.zeros((1, len(freqs)))
        rfi_V = np.zeros((1, len(freqs)))

        E = pol_beam(auto_beam=sinc_beam, cross_beam=None, params=[2., freqs],
                     ang_sep=np.mean(seps, keepdims=True))
        B = brightness_matrix(rfi_I, rfi_Q, rfi_U, rfi_V)
        V = RIME(B, E, delays[:,None], freqs)
        red_V[i] = avg_redundant_and_reshape(tele, V)

    return red_V
