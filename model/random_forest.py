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

for songs in ts.columns:
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state)
    val = ts[songs].values
    train = np.array([val[0:n_lags]])
    y = np.array(val[n_lags])
    for i in np.arange(1, np.size(val) - n_lags):
        train = np.concatenate((train, [val[i:i + n_lags]]))
        y = np.append(y, val[i + n_lags])
    clf.fit(train, y)
    for i in np.arange(pred_size):
        val = np.append(val, clf.predict([val[-n_lags:]]))
    pred[songs] = val[-pred_size:]

pred.to_csv('pred.csv', index = False)