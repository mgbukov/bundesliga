import pandas as pd
import numpy as np
import tensorflow as tf

#tf.set_random_seed(1337)
np.random.seed(1337)

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

dropout_vec=[0.3,0.4,0.5,0.6,0.7]

dim1_vec=[50]
dim2_vec=[500]



nb_epoch=100 #epoch_vec[int(sys.argv[1])-1] 
batch_size=10 #batch_vec[int(sys.argv[2])-1] 
dropout_p=0.4 #dropout_vec[int(sys.argv[3])-1] 

dense_dim1=50 # 50
dense_dim2=500 # 500

season15_end = 2752 # [1832 ,2139, 2445, 2751, 3057]


feats_import = pd.read_csv('All_Data_2006_2016.csv')
try:
    feats_import = feats_import.drop(['Unnamed: 0'], axis=1)
    print "Reshape successful"
except:
    print "Successful import"

# Goal differential Cut-off.
cutoff_GD = 3

feats = feats_import.drop(['Season', 'Gameday', 'TID_H', 'TID_A', 'Odds'], axis=1).drop('FTGD', axis=1)
feats.loc[:, 'HTGD'] = feats_import.loc[:, 'HTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD

label = feats_import.loc[:, 'FTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD

GD_min = -cutoff_GD
GD_max = +cutoff_GD
GD_spread = GD_max-GD_min + 1


# Different norms because I played around with different columns
norm = [9, 5, 36, 36, 36, 33, 35, 35, 20, 20, 11, 11, 4, 4, 1, 1, GD_spread-1, 1, 1]


feats = feats/norm

ID = np.eye(GD_spread)

X = feats.iloc[:season15_end].as_matrix()
X_test = feats.iloc[season15_end+1:].as_matrix()

y_pre = map(int, label.iloc[:season15_end].as_matrix())
y = np.array([ID[i] for i in y_pre])

y_pre2= map(int, label.iloc[season15_end+1:].as_matrix())
y_test= np.array([ID[i] for i in y_pre2])


feats.head()


# Assign an expected score with probabilities
def exp_score(x):
    multiplier = np.array(range(-cutoff_GD, cutoff_GD+1))
    return np.sum(np.array(x) * multiplier)


# I tested these parameters a little bit; they seem to work nicely for now.
model = Sequential()
model.add(Dense(dense_dim1, input_dim=len(X[0]), init='lecun_uniform', activation='relu')) 
model.add(Dense(dense_dim2, activation='relu'))
model.add(Dense(dense_dim2, activation='relu'))
model.add(Dense(dense_dim1, activation='relu'))
model.add(Dropout(dropout_p))
model.add(Dense(2*cutoff_GD+1, activation='relu'))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=False)

print "keras prediction:", model.evaluate(X, y, batch_size=10, verbose=False)

predictions_test = model.predict_proba(X_test)
predictions = model.predict_proba(X)

def performance_fn(predictions,X,feats_impt,test=''):

	# Cross-Tabulation for Away-Win (-1), Draw (0) or Home-win (1)
	performance_df = pd.concat([feats_impt.apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)),
	        					pd.Series(data=map(np.round, map(exp_score, predictions)), name='EXP', index=feats_impt.index)], axis=1) 
	success_res_df = pd.crosstab(performance_df.loc[:, "FTGD"].apply(np.sign), performance_df.loc[:, "EXP"].apply(np.sign))

	HDA_success=round(100* np.trace(success_res_df)/len(X),2)
	print test+" success identifying H, D, A is " + str(HDA_success) + " percent"


	# Cross-Tabulation for exact goal differential
	success_df = pd.crosstab(performance_df.loc[:, "FTGD"], performance_df.loc[:, "EXP"])#, margins=True)
	exp_min = int(performance_df.loc[:, "EXP"].min())
	exp_max = int(performance_df.loc[:, "EXP"].max())

	GD_success=round(np.sum([100*success_df.ix[i,i] for i in range(exp_min,exp_max+1)])/len(X),2)
	print test+" success identifying GD is " + str(GD_success) + " percent"

	return HDA_success, GD_success


HDA_train,GD_train = performance_fn(predictions,X,feats_import.iloc[:season15_end,-5])
HDA_test,GD_test = performance_fn(predictions_test,X_test,feats_import.iloc[season15_end+1:,-5],test='test')

#print "relative errors (HDA,GD) = ",(HDA_test/HDA_train,GD_test/GD_train)

