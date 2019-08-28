import dask.array as da
import os

def write_to_zarr(array, name, config, time_axis, freq_axis):

    path = config['process']['output_dir']
    time_chunk = config['process']['time_chunk']
    freq_chunk = config['process']['freq_chunk']
    full_path = os.path.join(path, name)
    A = da.from_array(array, chunks={time_axis:time_chunk, freq_axis:freq_chunk})
    da.to_zarr(A, full_path, overwrite=True, compute=True,)
               # chunks={time_axis:time_chunk, freq_axis:freq_chunk})
    A = da.from_zarr(full_path)

    return A
