__author__ = 'Hatsuyuki'

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
from score import score

def summation(a):
    n = np.size(a)
    for i in np.arange(1, n):
        a[i] += a[i - 1]
    return a

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
ts.index = pd.to_datetime(ts.index)

validation = ts.iloc[-20:]
ts = ts.iloc[:-20]

date_range = pd.date_range(start = '20150831', end = '20151030', freq = 'D')
date_size = np.size(date_range)
pred = pd.DataFrame(np.zeros((date_size, np.size(ts.columns))), index = date_range, columns = ts.columns)

f = open(r'../log/ARIMA.log', 'w')

c = 0
for artist in ts:
    print 'The ' + str(c) + '-th artist'
    c += 1
    max_score = 0
    best_p = -1
    best_q = -1
    for p in range(1, 10):
        for q in range(5):
            print p, q
            try:
                arima_model = ARIMA(ts[artist], (p, 1, q))
                arima_res = arima_model.fit(trend = 'nc', disp = -1)
                val_pred = arima_res.predict(start = '20150811', end = '20150830')
                val_pred = np.array(val_pred)
                val_pred[0] += ts[artist].iloc[-1]
                val_pred = summation(val_pred)
                val_score = score(validation[artist].values.reshape((20, 1)), val_pred.reshape((20, 1)))
                print 'Validation_Score: ' + str(val_score)
                if val_score > max_score:
                    arima_model = ARIMA(pd.concat((ts[artist], validation[artist])), (p, 1, q))
                    arima_res = arima_model.fit(trend = 'nc', disp = -1)
                    x = arima_res.predict(start = '20150831', end = '20151030')
                    x = np.array(x)
                    x[0] += validation[artist].iloc[-1]
                    pred[artist] = summation(x)
                    max_score = val_score
                    best_p = p
                    best_q = q
            except:
                print 'Error'
    f.write(str(best_p) + ', ' + str(best_q) + '\n')
    print 'Best Param: ' + str(best_p) + ', ' + str(best_q)

f.close()
pred.to_csv(r'../data/pred_ARIMA.csv')