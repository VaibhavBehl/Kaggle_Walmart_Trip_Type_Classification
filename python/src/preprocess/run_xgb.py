'''
Created on Dec 14, 2015

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

skf = StratifiedKFold(Y, n_folds=3, random_state=app_random_state_value)
skfList = list(skf)
train_index, test_index = skfList[0]
XD = X#.todense()
xTr, xTe = XD[train_index], XD[test_index]
yTr, yTe = Y[train_index], Y[test_index]

xg_train = xgb.DMatrix( xTr, label=yTr)
xg_test = xgb.DMatrix(xTe, label=yTe)
param = {'max_depth':50, 'eta':0.4, 'silent':1, 'objective':'multi:softprob', 'max_delta_step':1,
         'num_class':len(le.classes_), 'eval_metric':'mlogloss', 'seed':app_random_state_value, 'nthread':4,
         'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':3, 'lambda':8, 'alpha':3, 'gamma':1}
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 40
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
yhP = bst.predict( xg_test );
yhTe = np.argmax(yhP, axis=1)
classes = le.transform(le.classes_).tolist()
##

accuracy_score(yTe, yhTe)
cross_entropy(yhP, yTe, classes)
cfmat = confusion_matrix(yTe, yhTe, labels=classes)
np.savetxt('cfmat.5.4.csv',cfmat,delimiter=',',fmt='%i')
plt.matshow(cfmat)
plt.colorbar()
precision_recall_fscore_support(yTe, yhTe, average=None, labels=classes)

#MLP
from sklearn.neural_network import MLPClassifier

#temp code for cosine distance b/w DD's
# DD
x_new = sparse.lil_matrix(sparse.csr_matrix(XD)[:,list(range(47,115))])
x_new_T = x_new.T
dist = pairwise_distances(x_new_T, metric="cosine")
np.savetxt('dist_manhattan_train(68).csv',dist,delimiter=',',fmt='%f')
# FLN
x_FLN = sparse.lil_matrix(sparse.csr_matrix(XD)[:,list(range(flnStart,nextStart))])
x_FLN_T = x_FLN.T
dist_FLN = pairwise_distances(x_FLN_T, metric="cosine")
np.savetxt('dist_FLN.csv',dist_FLN,delimiter=',',fmt='%f')

#temp code for sklearn's nn (unsupervised)
from scipy.sparse import vstack
ddStart=88
flnStart=156
upcStart=5351
nextStart=11710

XTR, YTR = get_data(fullTrainFile)
XTE, YTE = get_data(fullTestFile)
x_new_tr = sparse.lil_matrix(sparse.csr_matrix(XTR)[:,list(range(upcStart-1,nextStart-1))])
x_new_te = sparse.lil_matrix(sparse.csr_matrix(XTE)[:,list(range(upcStart-1,nextStart-1))])
x_new_stack_T = vstack([x_new_tr,x_new_te]).T ## see boundry elements- print(sparse.csr_matrix(x_new_stack_T)[0])
##
#x_new = sparse.lil_matrix(sparse.csr_matrix(XD)[:,list(range(47,115))])
#x_new_T = x_new.T ##
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path
import networkx as nx
import pickle
feat='upc'
k=3
nbrs = NearestNeighbors(n_neighbors=k+1,metric='cosine',algorithm='brute').fit(x_new_stack_T) # k=(n_neighbors-1) (first neighbour is 'v' itself)
#distances, indices = nbrs.kneighbors(x_new_T) # not directly needed, for now
knnmatrix = nbrs.kneighbors_graph(x_new_stack_T,mode='distance') # sparse matrix(68x68) with nearest KNeighbours for each of the 68 pt
knnmatrix.data[np.where(knnmatrix.data<0)]=0
sp = graph_shortest_path(knnmatrix,directed=False) # shortest-path-edge-weight from (v_i to v_j), (doc-https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/graph_shortest_path.pyx)
G = nx.Graph(knnmatrix)
spl = nx.shortest_path(G, weight='weight') # shortest-path dict-array from each v_i to v_j, do len(array) to find path-length
## spl = nx.shortest_path(G) # Without weight (just connections-1/0)
pickle.dump(knnmatrix,open('knn_'+feat+'_k_'+str(k)+'.pickle.dump','wb')) # used to smooth out features
pickle.dump(sp,open('sp_all_'+feat+'_k_'+str(k)+'.pickle.dump','wb'))
#np.savetxt('sp_all_'+feat+'_k_'+str(k)+'.np.save', sp)
pickle.dump(spl,open('spl_all_'+feat+'_k_'+str(k)+'.pickle.dump','wb'))
##
knnmatrix_all = pickle.load(open('knn_'+feat+'_k_'+str(k)+'.pickle.dump','rb'))
sp_all = pickle.load(open('sp_all_'+feat+'_k_'+str(k)+'.pickle.dump','rb'))
#sp_all = np.loadtxt('sp_all_'+feat+'_k_'+str(k)+'.np.save')
spl_all = pickle.load(open('spl_all_'+feat+'_k_'+str(k)+'.txt','rb'))
#
cosine_dist_all = pairwise_distances(x_new_stack_T, metric="cosine")
pickle.dump(cosine_dist_all,open('cosine_dist_all_'+feat+'.pickle.dump','wb'))

#for => dist = pairwise_distances(x_new_T, metric="cosine")
# GDist = nx.Graph(dist)
# spl_w_dist = nx.shortest_path(GDist, weight='weight') # path-length always 1 for 'dist'(cosine)



#temp code to count mis-match frequency
c=0
misDic = {}
for i,pr in enumerate(yhTe):
    ac = yTe[i]
    if ac!=pr:
        key='a'+str(ac)+',p'+str(pr)
        if key not in misDic:
            misDic[key] = 1
        else:
            misDic[key] +=1
sorted_misDic = sorted(misDic.items(), key=operator.itemgetter(1))