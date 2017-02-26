from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

##########################################################

# load data
feats_import = pd.read_csv('All_Data_2006_2016.csv')
try:
    feats_import = feats_import.drop(['Unnamed: 0'], axis=1)
    print "Reshape successful"
except:
    print "Successful import"
    
# Set a cutoff for the goal differential: 
# All wins/losses with more than 3 goals difference are counted as wins/losses with goal differential of 3
cutoff_GD = 3
GD_min = -cutoff_GD
GD_max = +cutoff_GD
GD_spread = GD_max-GD_min + 1

# Import features and drop data which are not relevant or too specific, like teams playing. 
feats = feats_import.drop(['Season', 'Gameday', u'Link', u'TID_H', u'TID_A', u'TName_H', u'TName_A', 'Odds'], axis=1).drop('FTGD', axis=1)
feats.loc[:, 'HTGD'] = feats_import.loc[:, 'HTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD


# Overview over the data set
feats_import.head()


# Normalize the data set to values between 0 and 1
norm = [GD_spread-1, 9, 5, 1, 1, 1, 22, 22, 36, 36, 36, 33, 35, 35, 20, 20, 11, 11, 4, 4, 1, 1, 1, 1, 1]
feats = feats/norm

# Identify the point where season 2015/16 ends and thus the most recent season begins
season15_end = feats_import[feats_import['Season']==2016].index[0]

# Create the labels for the goal differentials between -cutoff_GD and +cutoff_GD
label = feats_import.loc[:, 'FTGD'].apply(min, args=(cutoff_GD,)).apply(max, args=(-cutoff_GD,)) + cutoff_GD

# Create a set of vectors which serve as the goal differential identifier
ID = np.eye(GD_spread)

### train data
# Bring X and y into numpy format
X = feats.iloc[:season15_end].as_matrix()
y_pre = map(int, label.iloc[:season15_end].as_matrix())

# input and output dimensions
input_dim=X.shape[1]
output_dim = len(set(y_pre))
nb_classes = len(set(y_pre))

# categorical classification values
y = np_utils.to_categorical(y_pre, nb_classes)

### test data
# Data from the most recent season
X_test = feats.iloc[season15_end+1:].as_matrix()
y_test_pre = map(int, label.iloc[season15_end+1:].as_matrix())

# input and output dimensions
input_dim_test=X_test.shape[1]
output_dim_test = len(set(y_test_pre))
nb_classes_test = len(set(y_test_pre))

# categorical classification values
y_test = np_utils.to_categorical(y_test_pre, nb_classes_test)


########################

### build log-reg model (softmax)

log_reg = Sequential() 
log_reg.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
batch_size = 10
nb_epoch = 100


log_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
history = log_reg.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, y_test)) 
score = log_reg.evaluate(X_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])


"""
classes = log_reg.predict_classes(X_test, batch_size=batch_size)
proba = log_reg.predict_proba(X_test, batch_size=batch_size)

print(classes)
print(proba)
"""



