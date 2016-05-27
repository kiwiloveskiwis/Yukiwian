__author__ = 'Hatsuyuki'

import numpy as np

def diff(a):
    n = np.size(a)
    return a - np.append([0], a)[:n]

def summation(a):
    n = np.size(a)
    for i in np.arange(1, n):
        a[i] += a[i - 1]
    return a