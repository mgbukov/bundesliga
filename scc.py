import pandas as pd
import numpy as np
import tensorflow as tf

#tf.set_random_seed(1337)
#qnp.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import sys
import os
import cPickle

#from matplotlib import pyplot as plt

epoch_vec=[100]
batch_vec=[10]
dropout_vec=[.4]

dim1_vec=[50]
dim2_vec=[700]
dim3_vec=[100]



nb_epoch=epoch_vec[int(sys.argv[1])-1] 
batch_size=batch_vec[int(sys.argv[2])-1] 
dropout_p=dropout_vec[int(sys.argv[3])-1] 

dense_dim1=50 # 10 # 50
dense_dim2=500 # 20 # 500
dense_dim3=500 # 500
dense_dim4=50 # 50

feats_import = pd.read_csv('All_Data_2006_2016.csv', index_col=0)

# Goal differential Cut-off.
cutoff_GD = 3

feats = feats_import.drop(['Season', 'Gameday', 'TID_H', 'TID_A'], axis=1).drop('FTGD', axis=1)
feats.loc[:, 'HTGD'] = feats_import.loc[:, 'HTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD

label = feats_import.loc[:, 'FTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD

GD_min = -cutoff_GD
GD_max = +cutoff_GD
GD_spread = GD_max-GD_min + 1


# Different norms because I played around with different columns
norm =  [9, 5, 36, 36, 36, 33, 35, 35, 20, 20, 11, 11, 4, 4, 1, 1, GD_spread-1, 1, 1, 1]


feats = feats/norm


ID = np.eye(GD_spread)

X = feats.iloc[:3000].as_matrix()
y_pre = map(int, label.iloc[:3000].as_matrix())

y = np.array([ID[i] for i in y_pre])

feats.head()


# Assign an expected score with probabilities
def exp_score(x):
    multiplier = np.array(range(-cutoff_GD, cutoff_GD+1))
    return np.sum(np.array(x) * multiplier)

# Run Keras
np.random.seed(7)

# I tested these parameters a little bit; they seem to work nicely for know.
model = Sequential()
model.add(Dense(dense_dim1, input_dim=len(X[0]), init='lecun_uniform', activation='relu')) 
model.add(Dense(dense_dim2, activation='relu'))
model.add(Dense(dense_dim3, activation='relu'))
model.add(Dense(dense_dim4, activation='relu'))
model.add(Dropout(dropout_p))
model.add(Dense(2*cutoff_GD+1, activation='relu'))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=False)
print model.evaluate(X, y, batch_size=10, verbose=False)[1]

predictions = model.predict_proba(X)

# Cross-Tabulation for Away-Win (-1), Draw (0) or Home-win (1)
performance_df = pd.concat([
        feats_import.iloc[:3000,-5].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)),
        pd.Series(data=map(np.round, map(exp_score, predictions)), name='EXP', index=feats_import.iloc[:3000].index)], axis=1)
success_res_df = pd.crosstab(performance_df.loc[:, "FTGD"].apply(np.sign), performance_df.loc[:, "EXP"].apply(np.sign))

HDA_success=round(100* np.trace(success_res_df)/3000.,2)
print "Success identifying H, D, A is " + str(HDA_success) + " percent"


# Cross-Tabulation for exact goal differential
success_df = pd.crosstab(performance_df.loc[:, "FTGD"], performance_df.loc[:, "EXP"])#, margins=True)
exp_min = int(performance_df.loc[:, "EXP"].min())
exp_max = int(performance_df.loc[:, "EXP"].max())

GD_success=round(np.sum([100*success_df.ix[i,i] for i in range(exp_min,exp_max+1)])/3000.,2)
print "Success identifying GD is " + str(GD_success) + " percent"

