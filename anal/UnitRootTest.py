#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Kiwi'
import pandas as pd
import numpy as np
import statsmodels.api as sm

count = 0
def measure(array):
    global count
    UnitRootTestResult=sm.tsa.adfuller(array)
    print 'ADF: ', UnitRootTestResult[0]
    print 'P-VALUE: ', UnitRootTestResult[1]
    print 'Critical values:'
    for keys in UnitRootTestResult[4].keys():
        print keys , UnitRootTestResult[4][keys]
    if UnitRootTestResult[1] < 0.10:
        print 'Time series is stationary'
        count = count + 1
    else:
        print 'Time series is nonstationary'
    print '\n'

ts_for_artists = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
for artists in ts_for_artists:
    print 'Artists: ', artists
    measure(ts_for_artists[artists])

print count
# print ts_for_artists.shape
