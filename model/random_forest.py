__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 7

train = np.zeros((ts.shape[0] - n_lags, n_lags))
y = np.zeros((ts.shape[0] - n_lags,))

c = 0

for songs in ts.columns:
    print c
    c += 1
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = 2)
    val = ts[songs].values
    for i in np.arange(0, np.size(val) - n_lags):
        train[i]= val[i:i + n_lags]
        y[i] = val[i + n_lags]
    clf.fit(train, y)
    for i in np.arange(pred_size):
        y_t = clf.predict([val[-n_lags:]])
        val = np.append(val, y_t)
    pred[songs] = val[-pred_size:]
    print pred[songs]

pred.to_csv(r'../data/pred.csv')