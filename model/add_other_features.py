__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np

songs = pd.read_csv(r'../data/songs.csv')

songs = pd.concat((songs, pd.get_dummies(songs['lang'])), axis = 1)
songs = pd.concat((songs, pd.get_dummies(songs['gender'])), axis = 1)
del songs['lang']
del songs['gender']

songs.to_csv(r'../data/songs_with_dummies.csv', index = False)

