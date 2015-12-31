'''
Created on Dec 27, 2015

@author: vaibhav
'''

from sklearn.cross_validation import StratifiedKFold
from utility import get_data
from utility import app_random_state_value
from utility import cross_entropy
from utility import delete_rows_csr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
import operator
from scipy import sparse
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import time
import pickle
import datetime

fullTrainFile = 'C:/myD/workarea/KaggleWallmartWorkarea/kaggle_wallmart/python/src/preprocess/BOTH_output_files/train_svm_light.v5.4.BOTH.2015-12-26_16-48-50.txt'
fullTestFile = 'C:/myD/workarea/KaggleWallmartWorkarea/kaggle_wallmart/python/src/preprocess/BOTH_output_files/test_svm_light.v5.4.BOTH.2015-12-26_09-24-51.txt'
X, Y = get_data(fullTrainFile)
#X = sparse.csr_matrix(X)[:,list(range(0,155))]
#Y_14 = np.where(Y==14) # to delete TT_14 rows
#Y = np.delete(Y, Y_14, 0)
#X = delete_rows_csr(X,Y_14)
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y).astype(np.int32)
#X=X.astype(np.float32)

skf = StratifiedKFold(Y, n_folds=3, random_state=app_random_state_value,shuffle=True)
skfList = list(skf)
train_index, test_index = skfList[0]
#X=X[:,0:150]
XD = X#.todense()
xTr, xTe = XD[train_index], XD[test_index]
yTr, yTe = Y[train_index], Y[test_index]

encoh = preprocessing.OneHotEncoder()
encoh.fit(yTr.reshape(-1, 1))
yTr = encoh.transform(yTr.reshape(-1, 1)).toarray()


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from keras.layers.advanced_activations import PReLU


model = Sequential()

#xTr=xTr[:,0:150]
lddm = xTr.shape[1]
print('total features-'+str(lddm))
# " petite"
model.add(Dense(output_dim=lddm*(2/3), input_dim=lddm, init='he_uniform', W_regularizer=regularizers.l1(0.002)))
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
 
# model.add(Dense(output_dim=lddm/2, init="glorot_uniform"))
# model.add(PReLU())
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# 
# model.add(Dense(output_dim=lddm/8, init="glorot_uniform"))
# model.add(PReLU())
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
 
model.add(Dense(output_dim=lddm/3, init='he_uniform'))
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(output_dim=38, init='he_uniform'))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.5, decay=1e-5, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')#, class_mode='categorical')

model.fit(xTr.astype(np.float32), yTr.astype(np.int32), nb_epoch=50, batch_size=256, show_accuracy=True, validation_split=0.03, 
          shuffle=True)
proba = model.predict_proba(xTe.astype(np.float32), batch_size=256)

print(proba)
#print(proba.shape())

yhP = proba
yhTe = np.argmax(yhP, axis=1)
classes = le.transform(le.classes_).tolist()
##
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
ts = ts.replace(' ', '_').replace(':', '-')
pickle.dump(model,open('model_'+str(ts)+'.pickle.dump','wb'))
pickle.dump(proba,open('yhP_proba_'+str(ts)+'.pickle.dump','wb'))
pickle.dump(yhTe,open('yhTe_'+str(ts)+'.pickle.dump','wb'))
##
print('accuracy:'+str(accuracy_score(yTe, yhTe)))
print('log-loss:'+str(cross_entropy(yhP, yTe, classes)))
