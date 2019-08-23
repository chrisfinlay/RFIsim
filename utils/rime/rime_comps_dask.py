import numpy as np
import dask.array as da

def phase_delays(distances, freqs):
    """
    Calculate the phase delay terms, K.

    Parameters:
    -----------
    distances: np.array (n_time,n_ant,n_src)
        Distances between each source and antenna.
    freqs: np.array (n_freq,)
        Frequencies

    Returns:
    --------
    delays: np.array (n_time,n_freq,n_bl,n_src)
        Phase delay terms.
    """

    c = 2.99792458e8
    phases = 2.*np.pi*freqs[None,:,None,None]*distances[:,None,:,:]/c

    return da.exp(1.j*phases)

def pol_beam(auto_beam, cross_beam, params, ang_sep):
    """
    Calculate the attenuation due to the primary beam for a
    given angular separation from the pointing direction.

    Parameters:
    -----------
    auto_beam: func
        Function that computes the auto-polarization beam attenuation.
        auto_beam(ang_sep, params)
    cross_beam: func
        Function that computes the auto-polarization beam attenuation.
        cross_beam(ang_sep, params)
    params: array-like
        The parameters to be used in the evaluation of the beams.
    ang_sep: np.array (n_srcs,)
        The angular separation between the source and the pointing
        direction in degrees.

    Returns:
    --------
    pol_beam: np.array (2,2,n_srcs,n_freqs)
        The polarized beam attenuation. [[HH, HV], [VH, VV]].
    """

    HH = auto_beam(ang_sep, params)
    HV = np.zeros(HH.shape)
    VH = np.zeros(HH.shape)
    VV = auto_beam(ang_sep, params)

#     Currently assuming HH = VV and both HV and VH are 0
    pol_beam = da.from_array([[HH, HV],[VH, VV]])

    return pol_beam

def gains(G):

    return da.from_array([[G[...,0], G[...,1]],
                     [G[...,2], G[...,3]]])

def brightness_matrix(I, Q, U, V):
    """
    Generate a brightness matrix from Stokes parameters.

    Parameters:
    -----------
    I: np.array (n_srcs,n_freqs)
        Stokes I.
    Q: np.array (n_srcs,n_freqs)
        Stokes Q.
    U: np.array (n_srcs,n_freqs)
        Stokes U.
    V: np.array (n_srcs,n_freqs)
        Stokes V.

    Returns:
    --------
    B: np.array (2,2,n_srcs,n_freqs)
        The brightness matrix.
    """

    HH = I + Q
    HV = U + 1.j*V
    VH = U - 1.j*V
    VV = I - Q

    B = da.from_array([[HH, HV], [VH, VV]])

    return B

def RIME(B, K, E, G, autos=False):
    """
    Calculate the RIME for a given brightness matrix,
    phase delay matrix, direction-dependent and
    direction-independent effects.

    Parameters:
    -----------
    B: np.array (2,2,n_time,n_freq,n_src)
        The brightness tensor.
    K: np.array (n_time,n_freq,n_ant,n_src)
        The phase delay tensor.
    E: np.array (2,2,n_time,n_freq,n_ant,n_src)
        The direction dependent effects tensor.
    G: np.array (2,2,n_time,n_freq,n_ant)
        The direction independent effects.

    Returns:
    --------
    V: np.array (2,2,n_time,n_freq,n_bl)
        The visibilies tensor. n_bl = n_ant*(n_ant-1)/2
    """

    a1, a2 = np.triu_indices(G.shape[-1], 0 if autos else 1)

#     Source coherency tensor
    X = B[:,:,:,:,None,:]*K[None,None,:,:,a1,:]*np.conjugate(K[None,None,:,:,a2,:])

#     Apparent source coherency
    A = da.einsum(('iAtfbs,ABtfbs,jBtfbs->ijtfbs'), E[...,a1,:], X,
                  da.conj(E[...,a2,:]), optimize='optimal')

#     Visibilities
    V = da.einsum('iAtfb,ABtfbs,jBtfb->ijtfb', G[...,a1], A,
                  da.conj(G[...,a2]), optimize='optimal')

    return V
