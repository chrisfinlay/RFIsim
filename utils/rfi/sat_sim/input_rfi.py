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
