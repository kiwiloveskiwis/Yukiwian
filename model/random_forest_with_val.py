__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from score import score
from create_submission import create_submission

random_state = 2016

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)

train_range = pd.date_range('20150301', '20150630', freq = 'D').astype('str')
valid_range = pd.date_range('20150701', '20150830', freq = 'D').astype('str')
valid_size = np.size(valid_range)
valid = pd.DataFrame(data = np.zeros((valid_size, np.size(ts.columns))), index = valid_range, columns = ts.columns)
ts_for_artists = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)

n_lags = 7

c = 0
for songs in ts.columns:
    c += 1
    if c % 100 == 0: print c
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state)
    val = ts.loc[train_range, songs].values
    train = np.array([val[0:n_lags]])
    y = np.array(val[n_lags])
    for i in np.arange(1, np.size(val) - n_lags):
        train = np.concatenate((train, [val[i:i + n_lags]]))
        y = np.append(y, val[i + n_lags])
    clf.fit(train, y)
    for i in np.arange(valid_size):
        val = np.append(val, clf.predict([val[-n_lags:]]))
    valid[songs] = val[-valid_size:]

valid.to_csv(r'../data/valid.csv', index = False)

sco = score(create_submission(pred = valid).values, ts_for_artists.loc[valid_range].values)
print sco