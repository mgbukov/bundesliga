import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

import pandas as pd



# read off data
Data = pd.read_csv('Bundesliga_Data_2006_2016.csv', index_col=0)
# create features data frame
Feats=pd.DataFrame(index=Data.index, columns=[])
Train=pd.DataFrame(index=Data.index, columns=[])
Bets =pd.DataFrame(index=Data.index, columns=[])

# ignore season-season correlations [drop year of game], but keep in-season correlations

# turn coeffs B365 into probabilities by L1-normalising vector [B365H,B365D,B365A]
norm_bets=np.sum(Data[['B365H','B365D','B365A']],axis=1)
Bets['B365H']= Data['B365H']/norm_bets
Bets['B365D']= Data['B365D']/norm_bets
Bets['B365A']= Data['B365A']/norm_bets

### normalise data wrt:

# Gameday: games in a season
Feats['Gameday']=Data['Gameday']/Data['Gameday'].max()
# TID_H/A: drop (for now), normalise wrt all seasons
Feats['TID_H']=Data['TID_H']/max(Data['TID_H'])
Feats['TID_A']=Data['TID_A']/max(Data['TID_A'])
# HTHG/AG: normalise wrt all-time max
Feats['HTHG']=Data['HTHG']/max(Data['HTHG'])
Feats['HTAG']=Data['HTAG']/max(Data['HTAG'])
# HTR: make binary
Feats['HTR']=Data['HTR'].replace(to_replace=['A','D','H'], value=[1.0,0.5,0.0])
# H/AS: normalise all time
Feats['HS']=Data['HS']/Data['HS'].max()
Feats['AS']=Data['AS']/Data['AS'].max()
# H/AST: give as percentage from normalised H/AS
Feats['HST']=Data['HST']/Data['HS']
Feats['AST']=Data['AST']/Data['AS']
# H/AF: normalise all time
Feats['HF']=Data['HF']/Data['HF'].max()
Feats['AF']=Data['AF']/Data['AF'].max()
# H/AC: normalise all time
Feats['HC']=Data['HC']/Data['HC'].max()
Feats['AC']=Data['AC']/Data['AC'].max()
# H/AY, H/AR: normalise all time
Feats['HY']=Data['HY']/Data['HY'].max()
Feats['AY']=Data['AY']/Data['AY'].max()
Feats['HR']=Data['HR']/Data['HR'].max()
Feats['AR']=Data['AR']/Data['AR'].max()

# FTHG/AG: use goal difference as a classifier
Train['FGD']=Data['FTHG']-Data['FTAG']
min_GD=int(min(Train['FGD']))
max_GD=int(max(Train['FGD']))
Id = np.eye(max_GD-min_GD)
Train_GD = np.array([Id[int(i)+min_GD,:] for i in Train['FGD'] ])


# FTR: use result as classifier
Train['FTR']=Data['FTR'].replace(to_replace=['H','D','A'], value=[0,1,2])
Id = np.eye(3)
Train_R = np.array([Id[i-1,:] for i in Train['FTR'] ])

print Train_R





