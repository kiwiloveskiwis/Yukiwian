__author__ = 'Hatsuyuki'

import numpy as np
import pandas as pd

def _sim(a, b):
    x = a[1].values
    y = b[1].values
    return np.mean(np.abs(x - y) / (x + y) / 2)

def read_sub(file_name):
    return pd.read_csv(r'../data/' + file_name, header = None)

def res_sim(a, b):
    return _sim(read_sub(a), read_sub(b))