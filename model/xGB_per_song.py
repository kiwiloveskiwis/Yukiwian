__author__ = 'Hatsuyuki'

import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime

random_state = 2016

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return (d2 - d1).days

params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eta": 0.1,
        "max_depth": 7,
        "silent": 0,
        "seed": random_state
}

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)

song = pd.read_csv(r'../data/songs.csv')
song = song.drop(['artist', 'lang', 'gender'], axis = 1)
song_dict = dict(zip(song.song.values, np.arange(song.shape[0])))

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 7

train = np.zeros((ts.shape[0] - n_lags, 2 + n_lags))
y = np.zeros((ts.shape[0] - n_lags, 1))

c = 0
row = 0
for songs in ts.columns:
    print c
    c += 1
    val = ts[songs].values
    publish_time, init_play = song.iloc[song_dict[songs]].drop('song').values
    publish_time = days_between(str(publish_time), '20150301')
    for i in np.arange(0, np.size(val) - n_lags):
        train[i] = np.append([publish_time, init_play], [val[i:i + n_lags]])
        y[i] = [val[i + n_lags]]
        publish_time += 1
        init_play += val[i]
    dat_train = xgb.DMatrix(train, label = y)
    eval = [(dat_train, 'train')]
    gbm = xgb.train(params, dat_train, 50, evals = eval, early_stopping_rounds = 10, verbose_eval = False)
    for i in np.arange(0, pred_size):
        val = np.append(val, gbm.predict(xgb.DMatrix(np.array([np.append([publish_time, init_play] , [val[-n_lags:]])])), ntree_limit = gbm.best_ntree_limit))
        publish_time += 1
        init_play += val[-n_lags - 1]
    pred[songs] = val[-pred_size:]
    #print pred[songs]

pred.to_csv(r'../data/pred_xgb_per_song.csv')
