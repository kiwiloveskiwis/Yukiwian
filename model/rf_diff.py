__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016

def diff(a):
    n = np.size(a)
    return a[1:n] - a[0:n - 1]

def summation(a):
    n = np.size(a)
    for i in np.arange(1, n):
        a[i] += a[i - 1]
    return a

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 7

c = 0

for songs in ts.columns:
    print c
    c += 1
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = 2)
    val = ts[songs].values
    init_val_test = val[-1]
    val = diff(val)
    train = np.array([val[0:n_lags]])
    y = np.array(val[n_lags])
    for i in np.arange(1, np.size(val) - n_lags):
        train = np.concatenate((train, [val[i:i + n_lags]]))
        y = np.append(y, val[i + n_lags])
    clf.fit(train, y)
    for i in np.arange(pred_size):
        val = np.append(val, clf.predict([val[-n_lags:]]))
    val[-pred_size] += init_val_test
    pred[songs] = summation(val[-pred_size:])

pred.to_csv(r'../data/pred_rf_diff.csv')