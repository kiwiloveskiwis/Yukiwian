__author__ = 'Hatsuyuki'

import numpy as np

def score(pred, act):
    (n, m) = pred.shape
    act = act + np.ones(act.shape)
    sig = np.sqrt(np.sum(((pred - act) / act) ** 2, axis = 0) / n)
    phi = np.sqrt(np.sum(act, axis = 0))
    f = np.dot(np.ones((m,)) - sig, phi)
    return f