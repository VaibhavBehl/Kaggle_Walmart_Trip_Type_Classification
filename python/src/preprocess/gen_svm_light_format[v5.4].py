'''
generate SVM LIGHT format for train and test data
'''

# import pydevd
# pydevd.settrace()
import collections
import csv
import datetime
import time
import numpy as np
from scipy import stats
import operator
from utility import get_sp_weights_from_matrix
from preprocess.utility import add_metrics_to_main_dict, get_sp_length_from_dict
import pickle

TRAIN_TT_IDX=0
TRAIN_VN_IDX=1
TRAIN_W_IDX=2
TRAIN_UPC_IDX=3
TRAIN_SC_IDX=4
TRAIN_DD_IDX=5
TRAIN_FLN_IDX=6
trainIdxDic = {'TT_IDX':TRAIN_TT_IDX,'VN_IDX':TRAIN_VN_IDX,'W_IDX':TRAIN_W_IDX,'UPC_IDX':TRAIN_UPC_IDX,
			'SC_IDX':TRAIN_SC_IDX,'DD_IDX':TRAIN_DD_IDX,'FLN_IDX':TRAIN_FLN_IDX}

TEST_VN_IDX=0
TEST_W_IDX=1
TEST_UPC_IDX=2
TEST_SC_IDX=3
TEST_DD_IDX=4
TEST_FLN_IDX=5
testIdxDic = {'VN_IDX':TEST_VN_IDX,'W_IDX':TEST_W_IDX,'UPC_IDX':TEST_UPC_IDX,
			'SC_IDX':TEST_SC_IDX,'DD_IDX':TEST_DD_IDX,'FLN_IDX':TEST_FLN_IDX}

run_for_test = True
#v1: baseline
#v2: separate DD for negative SC, no FLN when -ve SC(<<CHANGE in future, no -ve because of multinomial requirement)
#v3: added extra 7-8 features on top
#v4: *new* remove extra DD features and allow -ve summation in DD/FLN.
#starting with v5.4, can use the 'train' file generated from 'BOTH' mode separately
CURR_VER = 'v5.4'
MODE = 'ONLY'
if run_for_test:
	MODE = 'BOTH'
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
ts = ts.replace(' ', '_').replace(':', '-')

wOD=collections.OrderedDict()
ddOD=collections.OrderedDict()
flnOD=collections.OrderedDict()
upcOD=collections.OrderedDict()
odDic = {'wOD':wOD,'upcOD':upcOD,'ddOD':ddOD,'flnOD':flnOD}

def returnUpcKeys(Upc):
	retUpcKeys = []
	if len(Upc) == 11:
		retUpcKeys.append('UPC11_1_'+Upc[0])
		retUpcKeys.append('UPC11_5_'+Upc[1:6])
	elif len(Upc) == 10:
		retUpcKeys.append('UPC10_5_'+Upc[0:5])
	else:
		retUpcKeys.append('UPC_VAR_'+Upc)
	return retUpcKeys

def initDictionaries(fReader,LAST_ROW,idxDic,odDic):
	count = 0;
	next(fReader, None) # skip header
	for row in fReader:
		count = count + 1
		if(count % 10000 == 0):
			print('count -> ' + str(count))
		if count == LAST_ROW:
			break
		#if int(row[idxDic['TT_IDX']]) == 14:
			#continue
		w = row[idxDic['W_IDX']]
		upc = row[idxDic['UPC_IDX']]
		dd = row[idxDic['DD_IDX']]
		fln = row[idxDic['FLN_IDX']]
		wOD = odDic['wOD']
		upcOD = odDic['upcOD']
		ddOD = odDic['ddOD']
		flnOD = odDic['flnOD']
		if w not in wOD:
			wOD[w] = 1
		if upc.isdigit():
			upcKeys = returnUpcKeys(upc)
			for ke in upcKeys:
				if ke not in upcOD:
					upcOD[ke] = 1
		if dd == 'MENSWEAR':
			dd = 'MENS WEAR'
		if dd not in ddOD:
			ddOD[dd] = 1
		if fln.isdigit() and fln not in flnOD:
			flnOD[fln] = 1

#TRAIN
print('Init dictionaries from train')
TRAIN_LAST_ROW = 647055 #HARDCODED VALUE!!!!
TEST_LAST_ROW = 653647 #HARDCODED VALUE!!!!
trainFile = '../../../data_CSV/train.csv'
testFile = '../../../data_CSV/test.csv'
trainF = open(trainFile, 'r')
trainFReader = csv.reader(trainF)
initDictionaries(trainFReader,TRAIN_LAST_ROW,trainIdxDic,odDic)
#TEST
if run_for_test:
	testF = open(testFile, 'r')
	testFReader = csv.reader(testF)
	if False: #no init for TEST so as to not include unnecessary id's
		print('Init dictionaries from test')
		initDictionaries(testFReader,TEST_LAST_ROW,testIdxDic,odDic)

extraMetricsList = ['records', 'uniqDD', 'uniqFLN', 'sumSC', 
				'meanSC','medianSC','maxSC', 'sdSC', 'skewSC', 'kurtSC', 'iqrSC', 
				'meanDDSC','medianDDSC','maxDDSC', 'sdDDSC', 'skewDDSC', 'kurtDDSC', 'iqrDDSC',
				'meanFLNSC','medianFLNSC','maxFLNSC', 'sdFLNSC', 'skewFLNSC', 'kurtFLNSC', 'iqrFLNSC',
				'DDMaxSC1', 'DDMaxSC2',
				'sdUpcLen','skewUpcLen','kurtUpcLen']
upcLenList = [0,3,4,5,7,8,9,10,11,12]
spMetricList = ['sumSP','meanSP','medianSP','maxSP', 'sdSP', 'skewSP', 'kurtSP', 'iqrSP'] # len=8
totalSpUsed = 10
cosine_dist_all_dd = np.loadtxt('sp_files/dd/cosine_dist_all_dd.np.save')
sp_all_dd_k_1 = np.loadtxt('sp_files/dd/sp_all_dd_k_1.np.save')
spl_all_dd_k_1 = pickle.load(open('sp_files/dd/spl_all_dd_k_1.pickle.dump','rb'))
sp_all_dd_k_3 = np.loadtxt('sp_files/dd/sp_all_dd_k_3.np.save')
spl_all_dd_k_3 = pickle.load(open('sp_files/dd/spl_all_dd_k_3.pickle.dump','rb'))
cosine_dist_all_upc = pickle.load(open('sp_files/upc/cosine_dist_all_upc.pickle.dump','rb'))
sp_all_upc_k_1 = pickle.load(open('sp_files/upc/sp_all_upc_k_1.pickle.dump','rb'))
spl_all_upc_k_1 = pickle.load(open('sp_files/upc/spl_all_upc_k_1.pickle.dump','rb'))
sp_all_upc_k_3 = pickle.load(open('sp_files/upc/sp_all_upc_k_3.pickle.dump','rb'))
spl_all_upc_k_3 = pickle.load(open('sp_files/upc/spl_all_upc_k_3.pickle.dump','rb'))

knn_upc_k_3 = pickle.load(open('sp_files/upc/knn_upc_k_3.pickle.dump','rb'))

# SC after grouping by DD, for each VN, -> max, sd, skew, 
wList = list(wOD.keys())
upcList = list(upcOD.keys())
ddList = list(ddOD.keys())
flnList = list(flnOD.keys())
emStart = 1
upcLenStart = emStart + len(extraMetricsList)
spAllStart = upcLenStart + len(upcLenList)
wStart = spAllStart + totalSpUsed*len(spMetricList) # 'totalSpUsed' times # for all/sp(k=1/3)/spl(k=1/3)
ddStart = wStart + len(wList)
#ddNegStart = ddStart + len(ddList) #v2 changes
flnStart = ddStart + len(ddList)
upcStart = flnStart + len(flnList)
nextStart = upcStart + len(upcList)

#### Start Building SVMLight format file
def outputSvmFormatFile(fReader,LAST_ROW,outF,for_test,idxDic):
	count = 0
	preVN = -1
	firstCol = -1
	w = -1
	vnFeatDic = {}
	upcVnList = []
	scVnList = []
	ddVnList = []
	flnVnList = []
	next(fReader, None) # skip header
	for row in fReader:
		curVN = row[idxDic['VN_IDX']]
		count = count + 1
		if(count % 10000 == 0):
			print('--count -> ' + str(count))
			print('VisitNumber ->' + str(curVN))
			print(vnFeatDic)
		if((preVN != curVN and len(vnFeatDic)>0) or count == LAST_ROW):
			##  ************* CUSTOM FEATURES START  *************  ##
			vnFeatDic[emStart + extraMetricsList.index('records')] = len(ddVnList)
			if len(ddVnList)>1:
				vnFeatDic[emStart + extraMetricsList.index('uniqDD')] = len(np.unique(ddVnList))
				vnFeatDic[emStart + extraMetricsList.index('uniqFLN')] = len(np.unique(flnVnList))
			vnFeatDic[emStart + extraMetricsList.index('sumSC')] = sum(scVnList)
			# SC-VN Stat features
			if len(scVnList)>1:
				vnFeatDic[emStart + extraMetricsList.index('meanSC')] = np.mean(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('medianSC')] = np.median(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('maxSC')] = max(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('sdSC')] = np.std(scVnList,ddof=1)
				vnFeatDic[emStart + extraMetricsList.index('skewSC')] = stats.skew(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('kurtSC')] = stats.kurtosis(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('iqrSC')] = np.subtract(*np.percentile(scVnList, [75, 25]))
			# DD-VN-SC Stat features
			ddVnScList = [vnFeatDic.get(x) for x in list(range(ddStart,flnStart)) if vnFeatDic.get(x)!=None]
			if len(ddVnScList)>1:
				vnFeatDic[emStart + extraMetricsList.index('meanDDSC')] = np.mean(ddVnScList)
				vnFeatDic[emStart + extraMetricsList.index('medianDDSC')] = np.median(ddVnScList)
				vnFeatDic[emStart + extraMetricsList.index('maxDDSC')] = max(ddVnScList)
				vnFeatDic[emStart + extraMetricsList.index('sdDDSC')] = np.std(ddVnScList,ddof=1)
				vnFeatDic[emStart + extraMetricsList.index('skewDDSC')] = stats.skew(ddVnScList)
				vnFeatDic[emStart + extraMetricsList.index('kurtDDSC')] = stats.kurtosis(ddVnScList)
				vnFeatDic[emStart + extraMetricsList.index('iqrDDSC')] = np.subtract(*np.percentile(ddVnScList, [75, 25]))
			# FLN-VN Stat features
			flnVnScList = [vnFeatDic.get(x) for x in list(range(flnStart,upcStart)) if vnFeatDic.get(x)!=None]
			if len(flnVnScList)>1:
				vnFeatDic[emStart + extraMetricsList.index('meanFLNSC')] = np.mean(flnVnScList)
				vnFeatDic[emStart + extraMetricsList.index('medianFLNSC')] = np.median(flnVnScList)
				vnFeatDic[emStart + extraMetricsList.index('maxFLNSC')] = max(flnVnScList)
				vnFeatDic[emStart + extraMetricsList.index('sdFLNSC')] = np.std(flnVnScList,ddof=1)
				vnFeatDic[emStart + extraMetricsList.index('skewFLNSC')] = stats.skew(flnVnScList)
				vnFeatDic[emStart + extraMetricsList.index('kurtFLNSC')] = stats.kurtosis(flnVnScList)
				vnFeatDic[emStart + extraMetricsList.index('iqrFLNSC')] = np.subtract(*np.percentile(flnVnScList, [75, 25]))
			# DD's with top two SC; Cond: SC1>1 and SC2>SC1/3
			ddVnFeatDicSub = {k: vnFeatDic.get(k) for k in list(range(ddStart,flnStart)) if vnFeatDic.get(k)!=None}
			if len(ddVnFeatDicSub) > 1:
				sorted_ddVnFeatDicSub = sorted(ddVnFeatDicSub.items(), key=operator.itemgetter(1))
				ddVnMax1 = sorted_ddVnFeatDicSub[-1]
				if ddVnMax1[1] > 1: #  Cond1: SC1>1
					vnFeatDic[emStart + extraMetricsList.index('DDMaxSC1')] = ddVnMax1[0]
					if len(ddVnFeatDicSub) > 2:
						ddVnMax2 = sorted_ddVnFeatDicSub[-2]
						if ddVnMax2[1] > 1 and ddVnMax2[1] > ddVnMax1[1]/3: #  Cond2: SC2>1 and SC2>SC1/3
							vnFeatDic[emStart + extraMetricsList.index('DDMaxSC2')] = ddVnMax2[0]
			# DD adjacency usage
			ddVnFeatNewDic = {k-ddStart:v for k,v in ddVnFeatDicSub.items()} # sub ddStart from all keys of ddVnFeatDicSub
			cosine_dist_all_dd_list = get_sp_weights_from_matrix(ddVnFeatNewDic, cosine_dist_all_dd,multiply_10=True)
			add_metrics_to_main_dict(vnFeatDic, cosine_dist_all_dd_list, spAllStart+0*len(spMetricList), spMetricList)
			sp_all_dd_k_1_list = get_sp_weights_from_matrix(ddVnFeatNewDic, sp_all_dd_k_1)
			add_metrics_to_main_dict(vnFeatDic, sp_all_dd_k_1_list, spAllStart+1*len(spMetricList), spMetricList)
			sp_all_dd_k_3_list = get_sp_weights_from_matrix(ddVnFeatNewDic, sp_all_dd_k_3)
			add_metrics_to_main_dict(vnFeatDic, sp_all_dd_k_3_list, spAllStart+2*len(spMetricList), spMetricList)
			spl_all_dd_k_1_list = get_sp_length_from_dict(ddVnFeatNewDic, spl_all_dd_k_1)
			add_metrics_to_main_dict(vnFeatDic, spl_all_dd_k_1_list, spAllStart+3*len(spMetricList), spMetricList)
			spl_all_dd_k_3_list = get_sp_length_from_dict(ddVnFeatNewDic, spl_all_dd_k_3)
			add_metrics_to_main_dict(vnFeatDic, spl_all_dd_k_3_list, spAllStart+4*len(spMetricList), spMetricList)
			# UPC adjacency usage
			upcVnFeatDicSub = {k: vnFeatDic.get(k) for k in list(range(upcStart,nextStart)) if vnFeatDic.get(k)!=None}
			upcVnFeatNewDic = {k-upcStart:v for k,v in upcVnFeatDicSub.items()} # sub upcStart from all keys of upcVnFeatDicSub
			cosine_dist_all_upc_list = get_sp_weights_from_matrix(upcVnFeatNewDic, cosine_dist_all_upc,multiply_10=True)
			add_metrics_to_main_dict(vnFeatDic, cosine_dist_all_upc_list, spAllStart+5*len(spMetricList), spMetricList)
			sp_all_upc_k_1_list = get_sp_weights_from_matrix(upcVnFeatNewDic, sp_all_upc_k_1)
			add_metrics_to_main_dict(vnFeatDic, sp_all_upc_k_1_list, spAllStart+6*len(spMetricList), spMetricList)
			sp_all_upc_k_3_list = get_sp_weights_from_matrix(upcVnFeatNewDic, sp_all_upc_k_3)
			add_metrics_to_main_dict(vnFeatDic, sp_all_upc_k_3_list, spAllStart+7*len(spMetricList), spMetricList)
			spl_all_upc_k_1_list = get_sp_length_from_dict(upcVnFeatNewDic, spl_all_upc_k_1)
			add_metrics_to_main_dict(vnFeatDic, spl_all_upc_k_1_list, spAllStart+8*len(spMetricList), spMetricList)
			spl_all_upc_k_3_list = get_sp_length_from_dict(upcVnFeatNewDic, spl_all_upc_k_3)
			add_metrics_to_main_dict(vnFeatDic, spl_all_upc_k_3_list, spAllStart+9*len(spMetricList), spMetricList)
			# UPC add neighbors via lookup in dict- knn_upc_k_3 # knnmatrix[i].nonzero()[1] # nearest neighbors(k) for i'th element in matrix
			upcVnFeatDicUpdate={} # will hold original UPC and their KNeighbors
			for upcKey,upcVal in upcVnFeatNewDic.items():
				if upcVal>0:
					knn_idx_list = knn_upc_k_3[upcKey].nonzero()[1].tolist() # indexes of other UPC which are neighbor
					if upcKey in knn_idx_list:
						knn_idx_list.remove(upcKey)
					ll = len(knn_idx_list)
					if upcKey not in upcVnFeatDicUpdate:
						upcVnFeatDicUpdate[upcKey]=(1-ll/10)*upcVal #adding original member with 70% value
					else:
						upcVnFeatDicUpdate[upcKey]+=(1-ll/10)*upcVal
					# add neighbors(k) to upcVnFeatDicUpdate dictionary with a fraction of 'v' value
					for neigh in knn_idx_list:
						if neigh not in upcVnFeatDicUpdate:
							upcVnFeatDicUpdate[neigh]=0.1*upcVal
						else:
							upcVnFeatDicUpdate[neigh]+=0.1*upcVal
			upcVnFeatDicUpdate = {k+upcStart:v for k,v in upcVnFeatDicUpdate.items()}
			vnFeatDic.update(upcVnFeatDicUpdate)
			
			# 3 features for UPC length
			upcVnLenList = [len(j) for j in upcVnList]
			if len(upcVnLenList)>1:
				vnFeatDic[emStart + extraMetricsList.index('sdUpcLen')] = np.std(upcVnLenList,ddof=1)
				vnFeatDic[emStart + extraMetricsList.index('skewUpcLen')] = stats.skew(upcVnLenList)
				vnFeatDic[emStart + extraMetricsList.index('kurtUpcLen')] = stats.kurtosis(upcVnLenList)
			# upcLenList freq add
			if upcVnLenList:
				upcLenDic = {(upcLenStart + upcLenList.index(i)):upcVnLenList.count(i) for i in set(upcVnLenList)}
				vnFeatDic.update(upcLenDic)
			## ************* CUSTOM FEATURES END  *************  ##
			
			vnFeatDic[wStart + wList.index(w)] = 1 # WEEKDAY
			#Start writing the 'vnFeatDic' to the output file
			value = ['']
			vnFeatDicSorted = collections.OrderedDict(sorted(vnFeatDic.items()))
			for kk, vv in vnFeatDicSorted.items():
				value.append(' %s:%s' % (str(kk), str(vv)))
			outF.write(str(firstCol) + ''.join(value) + '\n');# Process the dict for VN and output a single line for that VN
			# reset for next VN
			vnFeatDic = {}
			upcVnList = []
			scVnList = []
			ddVnList = []
			flnVnList = []
		if count == LAST_ROW:
			break
		
		if for_test:
			firstCol = row[idxDic['VN_IDX']] # set VN for TEST
		else:	
			firstCol = row[idxDic['TT_IDX']] # set TT for TRAIN
		w = row[idxDic['W_IDX']] # doesn't change for a VN
		sc = int(row[idxDic['SC_IDX']])
		scVnList.append(sc)
		upc = row[idxDic['UPC_IDX']]
		if upc.isdigit():
			upcVnList.append(upc)
			upcKeys = returnUpcKeys(upc)
			for ke in upcKeys:
				if (not for_test) or (for_test and ke in upcList):
					keIndex = upcStart + upcList.index(ke)
					if keIndex not in vnFeatDic:
						vnFeatDic[keIndex] = sc
					else: 
						vnFeatDic[keIndex] += sc

		dd = row[idxDic['DD_IDX']]
		if dd == 'MENSWEAR':
			dd = 'MENS WEAR'
		ddVnList.append(dd)
		if (not for_test) or (for_test and dd in ddList):
			ddIdx = ddStart + ddList.index(dd)
			if ddIdx not in vnFeatDic:
				vnFeatDic[ddIdx] = sc
			else:
				vnFeatDic[ddIdx] += sc
		
		fln = row[idxDic['FLN_IDX']]
		if fln: #for empty fln # and sc>0:
			flnVnList.append(fln)
			if (not for_test) or (for_test and fln in flnList):
				flnIdx = flnStart + flnList.index(fln)
				if flnIdx not in vnFeatDic:
					vnFeatDic[flnIdx] = sc
				else:
					vnFeatDic[flnIdx] += sc
		
		preVN = curVN

print('dictionary init done')
print('Now converting train(many to one) to svm light(one to one) format')
outFileDir = MODE+'_output_files/'
trainOutF = open(outFileDir+'train_svm_light.'+CURR_VER+'.'+MODE+'.'+ts+'.txt', "w");
trainF.seek(0)
outputSvmFormatFile(trainFReader,TRAIN_LAST_ROW,trainOutF,False,trainIdxDic)
if run_for_test:
	testOutF = open(outFileDir+'test_svm_light.'+CURR_VER+'.'+MODE+'.'+ts+'.txt', "w");
	testF.seek(0)
	outputSvmFormatFile(testFReader,TEST_LAST_ROW,testOutF,True,testIdxDic)


#### Print all dictionary(wOD, ddOD, flnOD) to one file preserving index order
print('Print all dictionary to one file preserving index order')
dictWriter = csv.writer(open(outFileDir+'dictionary.'+CURR_VER+'.'+MODE+'.'+ts+'.csv', 'w', newline=''))
for em in extraMetricsList:
	dictWriter.writerow([emStart+extraMetricsList.index(em), em])
for upcl in upcLenList:
	dictWriter.writerow([upcLenStart+upcLenList.index(upcl), upcl])

for ii in range(0,totalSpUsed):
	for spml in spMetricList:
		dictWriter.writerow([spAllStart+ii*len(spMetricList)+spMetricList.index(spml), spml, ii])
# for spml in spMetricList:
# 	dictWriter.writerow([spAllStart+1*len(spMetricList)+spMetricList.index(spml), spml,1])
# for spml in spMetricList:
# 	dictWriter.writerow([spAllStart+2*len(spMetricList)+spMetricList.index(spml), spml,2])
# for spml in spMetricList:
# 	dictWriter.writerow([spAllStart+3*len(spMetricList)+spMetricList.index(spml), spml,3])
# for spml in spMetricList:
# 	dictWriter.writerow([spAllStart+4*len(spMetricList)+spMetricList.index(spml), spml,4])

for w in wList:
	dictWriter.writerow([wStart+wList.index(w), w])
for dd in ddList:
	dictWriter.writerow([ddStart+ddList.index(dd), dd])
#for dd in ddList:
	#dictWriter.writerow([ddNegStart+ddList.index(dd), dd])
for fln in flnList:
	dictWriter.writerow([flnStart+flnList.index(fln), fln])
for upc in upcList:
	dictWriter.writerow([upcStart+upcList.index(upc), upc])

dictWriter.writerow('*******')
dictWriter.writerow(['emStart', emStart])
dictWriter.writerow(['upcLenStart', upcLenStart])
dictWriter.writerow(['spAllStart', spAllStart])
dictWriter.writerow(['wStart', wStart])
dictWriter.writerow(['ddStart', ddStart])
dictWriter.writerow(['flnStart', flnStart])
dictWriter.writerow(['upcStart', upcStart])
dictWriter.writerow(['nextStart', nextStart])

#####################
# count = 0
# first = True;
# with open(sys.argv[1]) as fh:
# 	for line in fh:
# 		count = count + 1;
# 		if first == True:
# 			first = False;
# 			continue;
# 		fields = line.strip().split(",");
# 		row_id = int(fields[0]);  # Id field
# 		if(count % 100000 == 0 or count>8022700):
# 			print('count -> ' + str(count));
# 			print('Row Id->' + str(row_id))
# 		if(not row_id == curid or count == 8022757):  # hardcoded last row value
# 			if(len(data) == 0):
# 				trainOutF.write(str(curid) + ",0.0\n");
# 			else: 
# 				data.sort(key=lambda x:x[0]);
# 				minutes = map(lambda x: x[0], data);
# 				ref = map(lambda x: x[1], data);
# 				#last = 60 - minutes[-1];
# 				minuteList = list(minutes);
# 				valid_time = [0] * len(minuteList);
# 				valid_time[0] = minuteList[0];
# 				for n in range(1,len(minuteList)):
# 					valid_time[n] = minuteList[n] - minuteList[n-1];
# 				valid_time[-1] = valid_time[-1] + 60 - sum(valid_time);
# 				total_rain = 0;
# 				for dbz, hours in zip(ref, valid_time):
# 					#if(dbz > 0.0):
# 					if dbz not in (None, ""):
# 						total_rain = total_rain +  ((pow(pow(10, dbz/10)/200, 0.625)) * hours /60.0);
# 				trainOutF.write(",".join([str(curid),str(total_rain)]) + "\n");
# 			data = [];
# 			curid = row_id;	
# 		minutes = int(fields[1]);
# 		if len(fields[3]) == 0:  # Ref field
# 			ref = None;
# 		else:
# 			ref = float(fields[3]);
# 		data.append((minutes, ref));
