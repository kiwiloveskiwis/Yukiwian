__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np

songs = pd.read_csv(r'../data/songs.csv', index_col = 0)
artist = np.unique(songs['artist'].values)
ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
ts_for_art = pd.DataFrame(data = np.zeros((np.size(ts.index), np.size(artist))), index = ts.index, columns = artist)

c = 0
for col in ts:
    c += 1
    print c
    ts_for_art[songs.loc[col]['artist']] += ts[col]

ts_for_art.to_csv(r'../data/ts_for_artists.csv')