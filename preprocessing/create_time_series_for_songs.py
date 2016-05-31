__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np

songs = pd.read_csv(r'../data/songs.csv')
act = pd.read_csv(r'../data/act.csv')

date_range = pd.date_range('20150301', '20150830', freq = 'D')
num_days = np.size(date_range)
num_songs = songs.shape[0]

#ts = pd.DataFrame(data = np.zeros((num_days, num_songs)), index = date_range, columns = songs['song'])
ts_down = pd.DataFrame(data = np.zeros((num_days, num_songs)), index = date_range, columns = songs['song'])
ts_like = pd.DataFrame(data = np.zeros((num_days, num_songs)), index = date_range, columns = songs['song'])

for index, row in act.iterrows():
    print index
    #if row['action_type'] == 1: ts.loc[str(row['date']), row['song']] += 1
    if row['action_type'] == 2: ts_down.loc[str(row['date']), row['song']] += 1
    if row['action_type'] == 3: ts_like.loc[str(row['date']), row['song']] += 1

#ts.to_csv(r'../data/ts.csv')
ts_down.to_csv(r'../data/ts_down.csv')
ts_like.to_csv(r'../data/ts_like.csv')
