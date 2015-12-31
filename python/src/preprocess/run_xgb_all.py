'''
Created on Dec 14, 2015

@author: vaibhav
'''
'''
Created on Dec 14, 2015

@author: vaibhav
'''
from utility import get_data
from utility import app_random_state_value
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import time
import datetime

bothDataDir = 'C:/myD/workarea/KaggleWallmartWorkarea/kaggle_wallmart/python/src/preprocess/BOTH_output_files/'
fullTrainFile = bothDataDir+'train_svm_light.v5.4.BOTH.2015-12-26_16-48-50.txt'
xTr, yTr = get_data(fullTrainFile)
le = preprocessing.LabelEncoder()
le.fit(yTr)
yTr = le.transform(yTr) 
fullTestFile = bothDataDir+'test_svm_light.v5.4.BOTH.2015-12-26_16-48-50.txt'
xTe, vnTe = get_data(fullTestFile) # vnTe is visit number for TEST FILE


xg_train = xgb.DMatrix( xTr, label=yTr)
xg_test = xgb.DMatrix(xTe)#, label=yTe)
param = {'max_depth':50, 'eta':0.15, 'silent':1, 'objective':'multi:softprob', 'max_delta_step':1,
         'num_class':len(le.classes_), 'eval_metric':'mlogloss', 'seed':54325, 'nthread':4,
         'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':2, 'lambda':8, 'alpha':3, 'gamma':1}
# param = {'max_depth':50, 'eta':0.1, 'silent':1, 'objective':'multi:softprob', 
#          'num_class':38, "eval_metric":"mlogloss", "seed":app_random_state_value}
watchlist = [ (xg_train,'train') ]#, (xg_test, 'test') ]
num_round = 300
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
yhP = bst.predict( xg_test )
#yhTe = np.argmax(yhP, axis=1)
origClasses = le.classes_.tolist()
##

headers = ['TripType_' + str(int(oc)) for oc in origClasses]
headers.insert(0, 'VisitNumber')
headString = ','.join(headers)
#vnTe = list(range(1,len(yhP)+1)) #TEMP
vnTeInt = vnTe.astype(int)
yhPAll = np.c_[vnTeInt,yhP]


# print yhP to CSV file for FINAL Submission file
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
ts = ts.replace(' ', '_').replace(':', '-')
#dictWriter = csv.writer(open('Submission.'+ts+'.csv', 'w', newline=''))
np.savetxt(bothDataDir+'Submission.'+ts+'.csv', yhPAll, delimiter=',', header=headString, comments='')


