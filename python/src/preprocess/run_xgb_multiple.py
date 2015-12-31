'''
Created on Dec 25, 2015

@author: vaibhav
'''

from utility import get_data
from utility import app_random_state_value
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import time
import datetime
import random

bothDataDir = 'C:/myD/workarea/KaggleWallmartWorkarea/kaggle_wallmart/python/src/preprocess/BOTH_output_files/'
multipleYhpDir = bothDataDir + 'multiple_yhp_dir/'
fullTrainFile = bothDataDir+'train_svm_light.v5.3.BOTH.2015-12-25_00-03-42.txt'
xTr, yTr = get_data(fullTrainFile)
le = preprocessing.LabelEncoder()
le.fit(yTr)
yTr = le.transform(yTr) 
fullTestFile = bothDataDir+'test_svm_light.v5.3.BOTH.2015-12-25_00-03-42.txt'
xTe, vnTe = get_data(fullTestFile) # vnTe is visit number for TEST FILE

def return_xgb_result(xg_train,xg_test,paramDic):
    param = {'max_depth':50, 'eta':0.3, 'silent':1, 'objective':'multi:softprob',
             'num_class':len(le.classes_), 'eval_metric':'mlogloss', 'seed':paramDic['seed'], 'nthread':4,
             'max_delta_step':paramDic['max_delta_step'],'subsample':paramDic['subsample'], 
             'colsample_bytree':paramDic['colsample_bytree'], 'min_child_weight':paramDic['min_child_weight'], 
             'lambda':paramDic['lambda'], 'alpha':paramDic['alpha'], 'gamma':paramDic['gamma']}
    watchlist = [ (xg_train,'train') ]#, (xg_test, 'test') ]
    num_round = 45
    bst = xgb.train(param, xg_train, num_round, watchlist )
    yhP = bst.predict( xg_test )
    return yhP


xg_train = xgb.DMatrix(xTr, label=yTr)
xg_test = xgb.DMatrix(xTe)#, label=yTe)
yhPList = []
max_delta_step = [0,1,2,3,5]
subsample = [0.6,0.7,0.8,0.9,1]
colsample_bytree = [0.6,0.7,0.8,0.9,1]
min_child_weight = [1,2,3,10,15]
lambda_vals = [2,4,8,21,55]
alpha = [0,1,3,8,21]
gamma = [0,1,3,7,10]
for ii in range(0,20):
    seed = random.randrange(1,100000,1)
    randVals = [random.choice(max_delta_step),random.choice(subsample),random.choice(colsample_bytree)
                ,random.choice(min_child_weight), random.choice(lambda_vals), random.choice(alpha), random.choice(gamma)]
    paramDic = {'max_delta_step':randVals[0], 'subsample':randVals[1], 'colsample_bytree':randVals[2], 
                'min_child_weight':randVals[3], 'lambda':randVals[4], 'alpha':randVals[5], 'gamma':randVals[6],
                'seed':seed}
    print('In Round-'+str(ii)+',rVal-'+str(seed)+',params-'+str(paramDic))
    yhP = return_xgb_result(xg_train, xg_test, paramDic)
    yhPList.append(yhP)
    np.savetxt(multipleYhpDir + 'yhp_round'+str(ii)+'_('+ ','.join(str(x) for x in randVals) +').np.save', yhP)

yhPList = []
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round0_(5,0.6,1,3,2,0,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round1_(0,0.8,0.7,10,8,8,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round2_(0,0.9,1,10,55,8,1).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round3_(5,0.6,0.6,1,2,0,0).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round4_(2,1,0.9,1,4,8,10).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round5_(2,1,0.6,10,8,21,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round6_(5,0.6,0.9,1,8,0,1).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round7_(5,0.8,1,10,2,0,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round8_(3,0.6,0.8,15,55,0,10).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round9_(1,0.9,0.9,3,8,1,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round10_(5,0.9,0.8,3,2,21,0).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round11_(5,0.7,0.9,1,4,8,10).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round12_(2,0.8,1,10,4,3,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round13_(0,1,0.9,10,55,21,10).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round14_(2,1,1,3,21,8,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round15_(1,0.9,0.6,1,2,3,3).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round16_(1,0.6,0.7,3,55,8,1).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round17_(3,0.9,1,10,55,0,10).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round18_(3,0.6,0.9,3,21,8,0).np.save'))
yhPList.append(np.loadtxt(multipleYhpDir + 'yhp_round19_(0,0.8,0.9,3,55,1,1).np.save'))

yhPListLen = len(yhPList)
yhPSum = np.asmatrix(yhPList[0])
for ii in range(1,yhPListLen):
    yhPSum += np.asmatrix(yhPList[ii])
yhPavg = yhPSum/yhPListLen
yhPBest = np.loadtxt(multipleYhpDir + 'yhp_most_optimal.np.save')
yhPUpc = yhP
#yhPNew = (0.6*np.asmatrix(yhPUpc) + 0.4*np.asmatrix(yhPBest))
#yhP = yhPNew





## code for writing the submission file suing yhP
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
