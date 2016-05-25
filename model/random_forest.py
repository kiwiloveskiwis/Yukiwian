__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2006

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

for songs in ts.columns:
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state, verbose = 1)
    val = ts[songs].values
    train = np.array([val[0:7]])
    test = np.array(val[7])
    for i in np.arange(1, np.size(val) - 7):
        train = np.concatenate((train, [val[i:i + 7]]))
        test = np.append(test, val[i + 7])
    clf.fit(train, test)
    for i in np.arange(pred_size):
        val = np.append(val, clf.predict([val[-7:]]))
    pred[songs] = val[-pred_size:]
    #print np.shape(test)

pred.to_csv('pred.csv', index = False)