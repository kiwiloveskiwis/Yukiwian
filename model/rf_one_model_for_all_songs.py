__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
song = pd.read_csv(r'../data/songs_with_dummies.csv')
song = song.drop(['artist'], axis = 1)

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 7

train = []
y = []

c = 0
for songs in ts.columns:
    c += 1
    if c % 100 == 0: print c
    val = ts[songs].values
    for i in np.arange(0, np.size(val) - n_lags):
        train = np.concatenate((train, np.append(song.loc[song.song == songs].drop('song', axis = 1).values, [val[i:i + n_lags]])))
        y = np.append(y, val[i + n_lags])

print np.shape(train), np.shape(y)
clf = RandomForestRegressor(n_estimators = 500, random_state = random_state, verbose = 1)
clf.fit(train, y)

for songs in ts.columns:
    val = ts[songs].values
    for i in np.arange(0, pred_size):
        val = np.append(val, clf.predict(np.append(song.loc[song.song == songs].drop('song', axis = 1).values, [val[-n_lags:]])))
    pred[songs] = val[-pred_size:]

pred.to_csv('pred.csv', index = False)