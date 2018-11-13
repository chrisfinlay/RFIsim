import numpy as np
import pandas as pd


def inview(phase_centre, radius, min_flux):
    df = pd.read_csv('SUMSS_NVSS_Clean.csv')
    r = np.sqrt((df['RA']-phase_centre[0])**2 + (df['DEC']-phase_centre[1])**2)
    return df[(r<radius) & (df['Flux']>min_flux)]
