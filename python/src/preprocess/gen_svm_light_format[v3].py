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
CURR_VER = 'v3'
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

def initDictionaries(fReader,LAST_ROW,idxDic,odDic):
	count = 0;
	next(fReader, None) # skip header
	for row in fReader:
		count = count + 1
		if(count % 10000 == 0):
			print('count -> ' + str(count))
		if count == LAST_ROW:
			break
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
		if upc.isdigit() and upc not in upcOD:
			upcOD[upc] = 1
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
if run_for_test: #to not include unnecessary id's
	print('Init dictionaries from test')
	testF = open(testFile, 'r')
	testFReader = csv.reader(testF)
	initDictionaries(testFReader,TEST_LAST_ROW,testIdxDic,odDic)

extraMetricsList = ['records', 'uniqDD', 'sumSC', 'maxSC', 'sdSC', 'skewSC', 'kurtSC']
wList = list(wOD.keys())
#upcList = list(upcOD.keys())
ddList = list(ddOD.keys())
flnList = list(flnOD.keys())
emStart = 1
wStart = emStart + len(extraMetricsList)
ddStart = wStart + len(wList)
ddNegStart = ddStart + len(ddList) #v2 changes
flnStart = ddNegStart + len(ddList)
#upcStart = flnStart + len(flnList)

#### Start Building SVMLight format file
def outputSvmFormatFile(fReader,LAST_ROW,outF,for_test,idxDic):
	count = 0
	preVN = -1
	firstCol = -1
	w = -1
	vnFeatDic = {}
	scVnList = []
	ddVnList = []
	next(fReader, None) # skip header
	for row in fReader:
		curVN = row[idxDic['VN_IDX']]
		count = count + 1
		if(count % 10000 == 0):
			print('--count -> ' + str(count))
			print('VisitNumber ->' + str(curVN))
			print(vnFeatDic)
		if((preVN != curVN and len(vnFeatDic)>0) or count == LAST_ROW):
			vnFeatDic[emStart + extraMetricsList.index('records')] = len(ddVnList)
			vnFeatDic[emStart + extraMetricsList.index('uniqDD')] = len(np.unique(ddVnList))
			vnFeatDic[emStart + extraMetricsList.index('sumSC')] = sum(scVnList)
			vnFeatDic[emStart + extraMetricsList.index('maxSC')] = max(scVnList)
			if len(scVnList)>1:
				vnFeatDic[emStart + extraMetricsList.index('sdSC')] = np.std(scVnList,ddof=1)
				vnFeatDic[emStart + extraMetricsList.index('skewSC')] = stats.skew(scVnList)
				vnFeatDic[emStart + extraMetricsList.index('kurtSC')] = stats.kurtosis(scVnList)
			
			vnFeatDic[wStart + wList.index(w)] = 1 # add Weekday to vnFeatDic
			value = ['']
			vnFeatDicSorted = collections.OrderedDict(sorted(vnFeatDic.items()))
			for kk, vv in vnFeatDicSorted.items():
				value.append(' %s:%s' % (str(kk), str(vv)))
			outF.write(str(firstCol) + ''.join(value) + '\n');# Process the dict for VN and output a single line for that VN
			# reset it for next VN
			vnFeatDic = {}
			scVnList = []
			ddVnList = []
		if count == LAST_ROW:
			break
		
		if for_test:
			firstCol = row[idxDic['VN_IDX']] # set VN for TEST
		else:	
			firstCol = row[idxDic['TT_IDX']] # set TT for TRAIN
		w = row[idxDic['W_IDX']] # doesn't change for a VN
		#upc = row[idxDic['UPC_IDX']]
		sc = int(row[idxDic['SC_IDX']])
		scVnList.append(sc)
		ddVnList.append(row[idxDic['DD_IDX']])
		if sc>=0: #v2 changes
			ddIdx = ddStart + ddList.index(row[idxDic['DD_IDX']])
		else:
			ddIdx = ddNegStart + ddList.index(row[idxDic['DD_IDX']])
		if ddIdx not in vnFeatDic:
			vnFeatDic[ddIdx] = abs(sc)
		else:
			vnFeatDic[ddIdx] = vnFeatDic[ddIdx] + abs(sc)
		if row[idxDic['FLN_IDX']]: # and sc>0:
			flnIdx = flnStart + flnList.index(row[idxDic['FLN_IDX']])
			if flnIdx not in vnFeatDic:
				vnFeatDic[flnIdx] = sc
			else:
				vnFeatDic[flnIdx] = vnFeatDic[flnIdx] + sc
		
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
for w in wList:
	dictWriter.writerow([wStart+wList.index(w), w])
for dd in ddList:
	dictWriter.writerow([ddStart+ddList.index(dd), dd])
for dd in ddList:
	dictWriter.writerow([ddNegStart+ddList.index(dd), dd])
for fln in flnList:
	dictWriter.writerow([flnStart+flnList.index(fln), fln])


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
