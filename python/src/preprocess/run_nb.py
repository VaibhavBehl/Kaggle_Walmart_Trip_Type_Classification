'''
Created on Dec 12, 2015

@author: vaibhav
'''
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from utility import get_data
from utility import app_random_state_value
from sklearn.feature_extraction.text import TfidfTransformer
from utility import formatAndPrintMetrics
from utility import cross_entropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

fullTrainFile = 'C:/myD/workarea/KaggleWallmartWorkarea/kaggle_wallmart/data_CSV/train_svm_light.v2.new.txt'
X, Y = get_data(fullTrainFile)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y) 

skf = StratifiedKFold(Y, n_folds=3, random_state=app_random_state_value)
skfList = list(skf)
train_index, test_index = skfList[0]
XD = X#.todense()
xTr, xTe = XD[train_index], XD[test_index]
yTr, yTe = Y[train_index], Y[test_index]

clf = MultinomialNB()
clf = SGDClassifier(loss="hinge", penalty="l2")
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.8, max_depth=15, subsample=0.9, verbose=5, random_state=app_random_state_value)
clf = GaussianNB() #yhTeGNB = yhTe #yLogPGNB = yLogP
clf = RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=4, random_state=app_random_state_value)

#### Temp code to experiment on single class classification(binary-1/0)
yTrMod = [1 if a == 37 else 0 for a in yTr]
yTeMod = [1 if a == 37 else 0 for a in yTe]

xg_train = xgb.DMatrix( xTr, label=yTrMod)
xg_test = xgb.DMatrix(xTe, label=yTeMod)
param = {'max_depth':50, 'eta':0.1, 'silent':1, 'objective':'multi:softprob', 
         'num_class':2, "eval_metric":"mlogloss", "seed":app_random_state_value}
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
clf = xgb.train(param, xg_train, 100, watchlist )
yhPMod = clf.predict( xg_test );
yhTeMod = np.argmax(yhPMod, axis=1)

clf = clf.fit(xTr, yTrMod)
yhTeMod = clf.predict(xTe)
yhPMod = clf.predict_proba(xTe)

accuracy_score(yTeMod, yhTeMod)
precision_recall_fscore_support(yTeMod, yhTeMod, average=None, labels=np.unique(yTeMod))
#### Temp code end

clf = clf.fit(xTr, yTr)
yhTe = clf.predict(xTe)
yhP = clf.predict_proba(xTe)
classes = clf._classes
accuracy_score(yTe, yhTe)
cross_entropy(yhP, yTe, classes)
cfmat = confusion_matrix(yTe, yhTe, labels=classes)
plt.matshow(cfmat)
plt.colorbar()
precision_recall_fscore_support(yTe, yhTe, average=None, labels=classes)








maxIterations = 3 # number of CV results you want to see, value between 1 to 5
printPredictionToFile = False # deprecated.. should not use since we are doing CV, keep it false
algoVerbose = False
predFile = 'files/nb.mn.25.75.predictions'#sys.argv[2];
useTfIdf = False # this becomes feature 3! # True REDUCES MAE(good)
#Hyperparameters list
##
if(useTfIdf):
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X,Y)
scoreSumMatrix = np.zeros((4,len(classes)))
count=1
for train_index, test_index in skf:
    xTr, xTe = XD[train_index], XD[test_index]
    yTr, yTe = Y[train_index], Y[test_index]
    
    clf = GaussianNB()
    clf = clf.fit(xTr, yTr)
    yhTe = clf.predict(xTe)
    #yLogP = clf.predict_log_proba(xTe)
    
    #printMetrics(yTe,yhTe) #deprecated
    score = precision_recall_fscore_support(yTe, yhTe, average=None, labels=classes)
    scoreSumMatrix = scoreSumMatrix + score
    if(printPredictionToFile): # deprecated.. should not use
        # write yhTe to predictions file
        with open(predFile+str(count), 'w') as outRf:
            for yhVal in yhTe:
                outRf.write(str(int(yhVal))+'\n')
    count+=1
    if(count>maxIterations):
        formatAndPrintMetrics(scoreSumMatrix, maxIterations)
        break