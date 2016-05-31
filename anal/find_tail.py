__author__ = 'Kiwi and Hatsuyuki'

import pandas as pd
import numpy as np
import statsmodels.api as sm

count = 0

def measure(array):
    global count
    UnitRootTestResult=sm.tsa.adfuller(array, regression = 'ct')
    #print 'ADF: ', UnitRootTestResult[0]
    #print 'P-VALUE: ', UnitRootTestResult[1]
    #print 'Critical values:'
    #for keys in UnitRootTestResult[4].keys():
        #print keys , UnitRootTestResult[4][keys]
    if UnitRootTestResult[1] < 0.05:
        return True
    else: return False

ts_for_artists = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)

print 'artists,dlag'

c = 0

for artists in ts_for_artists:
    #print 'Artists: ', artists
    for dlag in np.arange(ts_for_artists.shape[0]):
        if measure(ts_for_artists[artists].iloc[dlag:]) == True:
            print artists + ',' + str(dlag)
            break
    c += 1
