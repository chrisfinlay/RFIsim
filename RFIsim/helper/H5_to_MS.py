
import argparse
import h5py
import os
import numpy as np
from casacore import tables as tb
from time import time
from astropy.time import Time

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H5_path", type=str,
                        help="Path to the H5 file created by RFIsim.")
    parser.add_argument("--MS_path", type=str,
                        help="Path to Measurement Set (MS) file you wish to write to.")

    return parser

def unix_to_MS_mjd(unix, offset=2):
    return Time(unix, format='unix').mjd*86400 + offset*3600

args = create_parser().parse_args()

fp = h5py.File(args.H5_path, 'r')
MS_path = args.MS_path

dirty = fp['/output/vis_dirty'].value
target = fp['/input/target'].value
UVW = fp['/input/UVW'].value
A1 = fp['/input/A1'].value
A2 = fp['/input/A2'].value
bandpass = fp['/input/bandpass'].value
freqs = fp['/input/frequencies'].value
times = fp['/input/unix_times'].value

n_time, n_bl, n_freq, n_pol = dirty.shape
n_ants = len(np.unique(np.concatenate([A1, A2])))
int_secs = 8

dicts = [{'NAME': 'dataHyperColumn',
  'SEQNR': 2,
  'SPEC': {'ActualMaxCacheSize': 0,
   'DEFAULTTILESHAPE': np.array([  4, 100, 100], dtype=np.int32),
   'HYPERCUBES': {'*1': {'BucketSize': 320000,
     'CellShape': np.array([   4, n_freq], dtype=np.int32),
     'CubeShape': np.array([     4,   n_freq, n_time*n_bl], dtype=np.int32),
     'ID': {},
     'TileShape': np.array([  4, 100, 100], dtype=np.int32)}},
   'IndexSize': 1,
   'MAXIMUMCACHESIZE': 0,
   'SEQNR': 2},
  'TYPE': 'TiledShapeStMan'},
 {'NAME': 'ModelDataColumn',
  'SEQNR': 4,
  'SPEC': {'ActualMaxCacheSize': 0,
   'DEFAULTTILESHAPE': np.array([  4, 100, 100], dtype=np.int32),
   'HYPERCUBES': {'*1': {'BucketSize': 320000,
     'CellShape': np.array([   4, n_freq], dtype=np.int32),
     'CubeShape': np.array([     4,   n_freq, n_time*n_bl], dtype=np.int32),
     'ID': {},
     'TileShape': np.array([  4, 100, 100], dtype=np.int32)}},
   'IndexSize': 1,
   'MAXIMUMCACHESIZE': 0,
   'SEQNR': 4},
  'TYPE': 'TiledShapeStMan'},
 {'NAME': 'CorrectedDataColumn',
  'SEQNR': 5,
  'SPEC': {'ActualMaxCacheSize': 0,
   'DEFAULTTILESHAPE': np.array([  4, 100, 100], dtype=np.int32),
   'HYPERCUBES': {'*1': {'BucketSize': 320000,
     'CellShape': np.array([   4, n_freq], dtype=np.int32),
     'CubeShape': np.array([     4,   n_freq, n_time*n_bl], dtype=np.int32),
     'ID': {},
     'TileShape': np.array([  4, 100, 100], dtype=np.int32)}},
   'IndexSize': 1,
   'MAXIMUMCACHESIZE': 0,
   'SEQNR': 5},
  'TYPE': 'TiledShapeStMan'}]

with tb.default_ms(MS_path) as ms:
    ms.addrows(n_time*n_bl)
    all_times = np.array([n_bl*[t] for t in unix_to_MS_mjd(times)]).flatten()

    ms.putcol('UVW', UVW)
    ms.putcol('FLAG', np.zeros((n_time*n_bl, n_freq, n_pol)).astype(bool))
    ms.putcol('WEIGHT', np.ones((n_time*n_bl, n_pol)).astype(np.float32))
    ms.putcol('SIGMA', np.ones((n_time*n_bl, n_pol)).astype(np.float32))
    ms.putcol('ANTENNA1', n_time*list(A1))
    ms.putcol('ANTENNA2', n_time*list(A2))
    ms.putcol('EXPOSURE', int_secs*np.ones(n_time*n_bl))
    ms.putcol('INTERVAL', int_secs*np.ones(n_time*n_bl))
    ms.putcol('SCAN_NUMBER', np.ones(n_time*n_bl))
    ms.putcol('TIME', all_times)
    ms.putcol('TIME_CENTROID', all_times)
    ms.addcols(tb.maketabdesc(tb.makearrcoldesc('DATA', 0., ndim=2,
                                                valuetype='complex')),
               dicts[0])
    ms.addcols(tb.maketabdesc(tb.makearrcoldesc('MODEL_DATA', 0., ndim=2,
                                                valuetype='complex')),
               dicts[1])
    ms.addcols(tb.maketabdesc(tb.makearrcoldesc('CORRECTED_DATA', 0., ndim=2,
                                                valuetype='complex')),
               dicts[2])

    ms.putcol('DATA', dirty.reshape(-1, 4096, 4))


with tb.table(os.path.join(MS_path, 'ANTENNA'), readonly=False) as ants:
    Ants = np.loadtxt('meerkat.itrf.txt', dtype='|S20')[:n_ants]

    ants.addrows(n_ants)
    ants.putcol('POSITION', Ants[:,:3].astype(np.float64))
    ants.putcol('TYPE', n_ants*['GROUND-BASED'])
    ants.putcol('DISH_DIAMETER', Ants[:,-3].astype(np.float64))
    ants.putcol('MOUNT', Ants[:,-2])
    ants.putcol('NAME', Ants[:,-2])


with tb.table(os.path.join(MS_path, 'DATA_DESCRIPTION'), readonly=False) as desc:
    desc.addrows(1)
    desc.putcol('FLAG_ROW', False)

with tb.table(os.path.join(MS_path, 'FEED'), readonly=False) as feed:
    feed.addrows(n_ants)

    feed.putcol('POSITION', np.zeros((n_ants, 3)))
    feed.putcol('BEAM_OFFSET', np.zeros((n_ants, 2, 2)))
    feed.putcol('POLARIZATION_TYPE', np.array([['X', 'Y'] for _ in range(n_ants)],
                                              dtype='|S1'))
    feed.putcol('POL_RESPONSE', np.ones((n_ants, 2, 2)))
    feed.putcol('RECEPTOR_ANGLE', np.zeros((n_ants, 2)))
    feed.putcol('ANTENNA_ID', np.arange(n_ants))
    feed.putcol('BEAM_ID', -1*np.ones(n_ants))
    feed.putcol('FEED_ID', np.zeros(n_ants))
    feed.putcol('INTERVAL', 1e30*np.ones(n_ants))
    feed.putcol('NUM_RECEPTORS', 2*np.ones(n_ants))

with tb.table(os.path.join(MS_path, 'FIELD'), readonly=False) as field:
    field.addrows(1)
    field.putcol('DELAY_DIR', np.deg2rad(target)[None,None,:])
    field.putcol('PHASE_DIR', np.deg2rad(target)[None,None,:])
    field.putcol('REFERENCE_DIR', np.deg2rad(target)[None,None,:])
    field.putcol('NAME', ['00'])


with tb.table(os.path.join(MS_path, 'OBSERVATION'), readonly=False) as obs:
    obs.addrows(1)
    now = Time(time(), format='unix').mjd*86400 + 7200
    obs.putcol('TIME_RANGE', np.array([[now-1800, now]]))
    obs.putcol('OBSERVER', ['RFIsim'])
    obs.putcol('PROJECT', ['RFIsim'])
    obs.putcol('RELEASE_DATE', [now])
    obs.putcol('TELESCOPE_NAME', ['MeerKAT'])

with tb.table(os.path.join(MS_path, 'POINTING'), readonly=False) as point:
    point.addrows(n_ants*n_time)
    now = n_ants*n_time*[Time(time(), format='unix').mjd*86400 + 7200]
    TARGET = np.ones((n_time*n_ants, 1, 1))*(np.deg2rad(target)[None,None,:])

    point.putcol('DIRECTION', TARGET)
    point.putcol('ANTENNA_ID', n_time*list(np.arange(n_ants)))
    point.putcol('INTERVAL', int_secs*np.ones(n_time*n_ants))
    point.putcol('NUM_POLY', np.zeros(n_ants*n_time))
    point.putcol('TARGET', TARGET)
    point.putcol('TIME', now)
    point.putcol('TIME_ORIGIN', now)
    point.putcol('TRACKING', n_ants*n_time*[True])

with tb.table(os.path.join(MS_path, 'POLARIZATION'), readonly=False) as pol:
    pol.addrows(1)

    pol.putcol('CORR_TYPE', np.arange(9, 13)[None,:])
    pol.putcol('CORR_PRODUCT', np.array([[[0, 0],[0, 1],[1, 0],[1, 1]]],
                                        dtype=np.int32))
    pol.putcol('FLAG_ROW', False)
    pol.putcol('NUM_CORR', 4)

with tb.table(os.path.join(MS_path, 'SPECTRAL_WINDOW'), readonly=False) as spec:
    spec.addrows(1)
    chan_width = np.array(n_freq*list(np.diff(freqs[:2])))[None,:]

    spec.putcol('MEAS_FREQ_REF', 5)
    spec.putcol('CHAN_FREQ', freqs[None,:])
    spec.putcol('REF_FREQUENCY', [freqs[0]])
    spec.putcol('CHAN_WIDTH', chan_width)
    spec.putcol('EFFECTIVE_BW', chan_width)
    spec.putcol('RESOLUTION', chan_width)
    spec.putcol('FLAG_ROW', False)
    spec.putcol('FREQ_GROUP', 0)
    spec.putcol('FREQ_GROUP_NAME', 'Group 1')
    spec.putcol('IF_CONV_CHAIN', 0)
    spec.putcol('NAME', '00')
    spec.putcol('NET_SIDEBAND', 1)
    spec.putcol('NUM_CHAN', n_freq)
    spec.putcol('TOTAL_BANDWIDTH', freqs[-1]-freqs[0])

with tb.table(os.path.join(MS_path, 'STATE'), readonly=False) as state:
    state.addrows(1)

    state.putcol('CAL', 0)
    state.putcol('FLAG_ROW', False)
    state.putcol('LOAD', 0)
    state.putcol('OBS_MODE', 'OBSERVE_TARGET.ON_SOURCE')
    state.putcol('REF', False)
    state.putcol('SIG', True)
    state.putcol('SUB_SCAN', 0)
