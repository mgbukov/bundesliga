from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# parameters
nb_epoch = 100
batch_size = 10
#
hid1_dim=200
hid2_dim=120
hid3_dim=50 




feats_import = pd.read_csv('All_Data_2006_2016.csv', index_col=0)

feats = feats_import.drop(['Season', 'Gameday', 'TID_H', 'TID_A', 'HY', 'AY', 'HR', 'AR','HF','AF'], axis=1).drop('FTGD', axis=1)
feats.loc[:, 'HTGD'] = feats.loc[:, 'HTGD'] - feats.loc[:, 'HTGD'].min()
feats = feats/feats.apply(max)


"""
GD_min = int(feats_import.loc[:, 'FTGD'].min())
GD_max = int(feats_import.loc[:, 'FTGD'].max())
label = feats_import.loc[:, 'FTGD'] - GD_min
"""

label = np.sign( feats_import.loc[:, 'FTGD'] )+1


GD_spread = 3 #GD_max-GD_min + 1
ID = np.eye(GD_spread)

X = feats.iloc[:2700].as_matrix()
y_pre = map(int, label.iloc[:2700].as_matrix())

y = np.array([ID[i] for i in y_pre])



neuron='sigmoid'
def baseline_model():
    model = Sequential()
    model.add(Dense(hid1_dim,input_dim=len(X[0]), init='lecun_uniform', activation=neuron))
    model.add(Dropout(0.5)) 
    model.add(Dense(hid2_dim, activation=neuron))
    model.add(Dropout(0.5))
    #model.add(Dense(hid3_dim, activation=neuron))
    #model.add(Dropout(0.5))
    #model.add(Dense(hid3_dim/2, activation=neuron))
    model.add(Dense(y.shape[1], activation='sigmoid'))
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam', #
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, 
							nb_epoch=nb_epoch, 
							batch_size=batch_size, 
							verbose=False)


model= baseline_model()
model.fit(X,y,batch_size=batch_size,nb_epoch=nb_epoch,verbose=False)
test=model.predict_proba(X)
print test[153], test[504], test[2016]

exit()

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, y)#, cv=kfold)
print "Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)



X_test = feats.iloc[2701:].as_matrix()
y_test_pre = map(int, label.iloc[2701:].as_matrix())

y_test = np.array([ID[i] for i in y_test_pre])

oos = cross_val_score(estimator, X_test, y_test, cv=kfold)

print "Baseline: %.2f%% (%.2f%%)" % (oos.mean()*100, oos.std()*100)

