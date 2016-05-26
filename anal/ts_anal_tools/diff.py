__author__ = 'Hatsuyuki'

import numpy as np

def diff(a):
    n = np.size(a)
    return a - np.append([0], a)[:n]
