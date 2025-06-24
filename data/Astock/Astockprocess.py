# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:20:19 2024

@author: 2507
"""

import pandas as pd
import unicodedata
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tqdm


# Load  dataset
path=r'../../data/Astock/'
train_data_final = pd.read_csv(path+'trainAstock.csv',sep='\t')
val_data_final = pd.read_csv(path+'valAstock.csv',sep='\t')
test_data_final = pd.read_csv(path+'testAstock.csv',sep='\t')


##萃取x變數和y變數
xtrain_bigdata = []
n=2
x_scaler = MinMaxScaler(feature_range = (-1, 1))
#使用x變數
trainx_variable=train_data_final.iloc[:,2:]
trainx_variable=pd.DataFrame(x_scaler.fit_transform(trainx_variable))
#使用y變數
trainy_variable=pd.DataFrame(train_data_final.iloc[:,1])

for i in tqdm.tqdm_notebook(range(0,len(trainx_variable)-n+1)):       
    x_trainadd=trainx_variable.iloc[i:i+n]. values
    xtrain_bigdata.append(np.transpose(x_trainadd))
    ##轉成array
print(xtrain_bigdata[0])


x_train=np.array(xtrain_bigdata)
y_train=np.array(train_data_final.iloc[1:,1])

#val
xval_bigdata = []
#使用x變數
valx_variable=val_data_final.iloc[:,2:]
trainx_variable=pd.DataFrame(x_scaler.fit_transform(valx_variable))
#使用y變數
valy_variable=pd.DataFrame(val_data_final.iloc[:,1])

for i in tqdm.tqdm_notebook(range(0,len(valx_variable)-n+1)):       
    x_valadd=valx_variable.iloc[i:i+n]. values
    xval_bigdata.append(np.transpose(x_valadd))  
    ##轉成array
print(xval_bigdata[0])

x_val=np.array(xval_bigdata)
y_val=np.array(val_data_final.iloc[1:,1])


#test
xtest_bigdata = []
#使用x變數
testx_variable=test_data_final.iloc[:,2:]
testx_variable=pd.DataFrame(x_scaler.fit_transform(testx_variable))
#使用y變數
testy_variable=pd.DataFrame(test_data_final.iloc[:,1])

for i in tqdm.tqdm_notebook(range(0,len(testx_variable)-n+1)):       
    x_testadd=testx_variable.iloc[i:i+n]. values
    xtest_bigdata.append(np.transpose(x_testadd))  
    ##轉成array
print(xtest_bigdata[0])

x_test=np.array(xtest_bigdata)
y_test=np.array(test_data_final.iloc[1:,1])

np.save(path+'Astockx_train',x_train)
np.save(path+'Astockx_val',x_val)
np.save(path+'Astockx_test',x_test)
np.save(path+'Astocky_train',y_train)
np.save(path+'Astocky_val',y_val)
np.save(path+'Astocky_test',y_test)

