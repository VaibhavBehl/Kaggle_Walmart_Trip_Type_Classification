'''
Created on Dec 12, 2015

@author: vaibhav
'''
import sys
import math
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
import heapq
from scipy import sparse
import collections
import matplotlib.pyplot as plt
from scipy import stats

# don't change this value in between testing diff algos. This value determines how the StratifiedKFold cuts All Data into Tr and Te
app_random_state_value = 1
classes = [3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]

def cross_entropy(predBak,label,classes,n=''):
    pred=np.copy(predBak)
    pred = np.asarray(pred)
    if n:
        predM = [heapq.nlargest(n, a)[n-1] for a in pred]
        mIdx = [a<predM[i] for i,a in enumerate(pred)]
        mIdx = np.asarray(mIdx)
        pred[mIdx]=0
        #pred[np.invert(mIdx)] = 1
    # TODO: pick top N and set them to 1, set rest to zero
    pred = pred / pred.sum(axis=1)[:, np.newaxis] # rescaling (scaling sum to 1 for all rows(axis=1))
    eps=1e-15 # cutoff
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean([np.log(p[ classes.index(label[i]) ]) for i, p in enumerate(pred)])

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    mat = sparse.csr_matrix(mat)
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def get_sp_weights_from_matrix(ddVnFeatNewDic, sp_all, multiply_10=False):
    """
    ddVnFeatNewDic : dictionary of DD/UPC as key and its count as value
    sp_all : matrix of DDxDD with edge weights
    return: a list of edge-weight values for all combinations of DD's  found in ddVnFeatNewDic
    """
    ddVnFeatNewKeyList = list(ddVnFeatNewDic.keys())
    sp_all_list = []
    for ii in range(0,len(ddVnFeatNewKeyList)):
        for jj in range(ii+1,len(ddVnFeatNewKeyList)):
            #print('ii:'+str(ii)+',jj:'+str(jj)) #iterate spMat/spLenMat and output a list for wts and len, on which to run stats
            ijWeight = (sp_all[ddVnFeatNewKeyList[ii]][ddVnFeatNewKeyList[jj]])
            if multiply_10:
                ijWeight = ijWeight*10
            ijWeight = ijWeight**2
            if ijWeight == 0:
                ijWeight = -1
            sp_all_list.append(ijWeight)
    return sp_all_list

def get_sp_length_from_dict(ddVnFeatNewDic, spl_all):
    """
    ddVnFeatNewDic : dictionary of DD as key and its count as value
    spl_all : a dict of dict with edge path for all DD pairs 
    return: a list of edge-path-length values for all combinations of DD's found in ddVnFeatNewDic
    """
    ddVnFeatNewKeyList = list(ddVnFeatNewDic.keys())
    spl_all_list = []
    for ii in range(0,len(ddVnFeatNewKeyList)):
        for jj in range(ii+1,len(ddVnFeatNewKeyList)):
            #print('ii:'+str(ii)+',jj:'+str(jj)) #iterate spMat/spLenMat and output a list for wts and len, on which to run stats
            if ddVnFeatNewKeyList[jj] in spl_all[ddVnFeatNewKeyList[ii]]:
                spl_all_list.append((len(spl_all[ddVnFeatNewKeyList[ii]][ddVnFeatNewKeyList[jj]])-1)**2)
            else:
                spl_all_list.append(-1)
    return spl_all_list

def add_metrics_to_main_dict(vnFeatDic, inputList, dicStartIndex, spMetricList):
    if len(inputList)>0:
        vnFeatDic[dicStartIndex + spMetricList.index('sumSP')] = sum(inputList)
    if len(inputList)>1:
        vnFeatDic[dicStartIndex + spMetricList.index('meanSP')] = np.mean(inputList)
        vnFeatDic[dicStartIndex + spMetricList.index('medianSP')] = np.median(inputList)
        vnFeatDic[dicStartIndex + spMetricList.index('maxSP')] = max(inputList)
        vnFeatDic[dicStartIndex + spMetricList.index('sdSP')] = np.std(inputList,ddof=1)
        vnFeatDic[dicStartIndex + spMetricList.index('skewSP')] = stats.skew(inputList)
        vnFeatDic[dicStartIndex + spMetricList.index('kurtSP')] = stats.kurtosis(inputList)
        vnFeatDic[dicStartIndex + spMetricList.index('iqrSP')] = np.subtract(*np.percentile(inputList, [75, 25]))

# def cross_entropy1(predBak,label,classes,n=''):
#     pred=np.copy(predBak)
#     pred = np.asarray(pred)
#     if n:
#         predM = [heapq.nlargest(n, a)[n-1] for a in pred]
#         mIdx = [a<predM[i] for i,a in enumerate(pred)]
#         mIdx = np.asarray(mIdx)
#         pred[mIdx]=0
#         #pred[np.invert(mIdx)] = 1
#     # TODO: pick top N and set them to 1, set rest to zero
#     pred = pred / pred.sum(axis=1)[:, np.newaxis] # rescaling (scaling sum to 1 for all rows(axis=1))
#     eps=1e-15 # cutoff
#     pred = np.clip(pred, eps, 1 - eps)
#     cEps = 0
#     logProbSum = 0
#     lessThanList = [] # x
#     y = []
#     for i, p in enumerate(pred):
#         iPredictedProb = p[classes.index(label[i])]
#         if iPredictedProb <= 1/38:
#             cEps += 1
#             lessThanList.append(int(label[i]))
#             y.append(np.log(iPredictedProb))
#             #print('iPredictedProb='+ str(iPredictedProb) +'yhPMod[i]='+str(yhPMod[i])+', yhTeMod[i]='+str(yhTeMod[i]))
#         logProbSum += np.log(iPredictedProb)
#     collections.Counter(lessThanList)
#     plt.scatter(lessThanList, y)
#     print('cEps='+str(cEps))
#     return -(logProbSum/len(pred))

def formatAndPrintMetrics(scoreSumMatrix, maxIterations):
    scoreSumMatrix = scoreSumMatrix/maxIterations
    scoreSumMatrix = scoreSumMatrix*100
    scoreSumMatrix = np.around(scoreSumMatrix.astype(np.double),4)
    print('average P=' + str(scoreSumMatrix[0]))
    print('average R=' + str(scoreSumMatrix[1]))
    print('average F=' + str(scoreSumMatrix[2]))

#this is only to generate for a single pair of (orig,pred), use formatAndPrintMetrics for others
def printMetricsPRF(origY, predY):
    
    if(len(origY) != len(predY)):
        sys.exit('Custom error msg: length of orig and pred Y not equal')
    
    belongInClass = [0,0,0,0,0]
    classifiedAsClass = [0,0,0,0,0]
    correctlyClassifiedAsClass = [0,0,0,0,0]
    
    mseSum = 0
    abSum = 0
    mseCount = 0
    for oy, py in zip(origY, predY):
        mseSum = mseSum + math.pow(oy-py,2)
        abSum = abSum + abs(oy-py)
        mseCount += 1
        for i,cc in enumerate(classes):
            if oy == cc:
                belongInClass[i] += 1
            if py == cc:
                classifiedAsClass[i] += 1
            if oy == py:
                if oy == cc:
                    correctlyClassifiedAsClass[i] += 1
    
    print('MSE=' + str(mseSum/mseCount))
    print('MAE=' + str(abSum/mseCount))
    for i,cc in enumerate(classes):
        pClass = round(100*correctlyClassifiedAsClass[i]/classifiedAsClass[i],4);
        rClass = round(100*correctlyClassifiedAsClass[i]/belongInClass[i],4);
        fClass = round(2*pClass*rClass/(pClass + rClass),4);
        print('class ' + str(cc) + ' P=' + str(pClass) + ' R=' + str(rClass) + ' F=' + str(fClass))

mem = Memory("./mycache_can_delete")
@mem.cache
def get_data(dataFile):
    data = load_svmlight_file(dataFile)
    return data[0], data[1]

