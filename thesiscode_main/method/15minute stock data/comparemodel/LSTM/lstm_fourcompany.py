# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:53:25 2024

@author: User
"""
####套件安裝###########
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import datetime
import requests
import json
import time
import csv
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median
from dateutil.relativedelta import relativedelta
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm 
import tensorflow as tf
import os
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout



#載入資料
AAPL=pd.read_csv( r'C:\Users\2507\Desktop\遠端資料\data\15mindata\AAPL\AAPL15min.csv')  
AAPL=AAPL.iloc[:,2:]
vol=AAPL['volume']
close=AAPL['close']
#volume和close欄位對調
AAPL['close']=vol
AAPL['volume']=close
AAPL.columns=['open', 'high', 'low', 'volume', 'close']
TSLA=pd.read_csv( r'C:\Users\2507\Desktop\遠端資料\data\15mindata\TSLA\TSLA15min.csv')  
TSLA=TSLA.iloc[:,2:]
vol=TSLA['volume']
close=TSLA['close']
#volume和close欄位對調
TSLA['close']=vol
TSLA['volume']=close
TSLA.columns=['open', 'high', 'low', 'volume', 'close']
MSFT=pd.read_csv( r'C:\Users\2507\Desktop\遠端資料\data\15mindata\MSFT\MSFT15min.csv')  
MSFT=MSFT.iloc[:,2:]
vol=MSFT['volume']
close=MSFT['close']
#volume和close欄位對調
MSFT['close']=vol
MSFT['volume']=close
MSFT.columns=['open', 'high', 'low', 'volume', 'close']
IBM=pd.read_csv( r'C:\Users\2507\Desktop\遠端資料\data\15mindata\IBM\IBM15min.csv')  
IBM=IBM.iloc[:,2:]
vol=IBM['volume']
close=IBM['close']
#volume和close欄位對調
IBM['close']=vol
IBM['volume']=close
IBM.columns=['open', 'high', 'low', 'volume', 'close']




final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)

stock_id=['AAPL','TSLA','MSFT','IBM']

stock_mae=[] 
stock=[] 

final_data_real_copy=final_data_real

x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))


for k in range(0,len(stock_id)):
    print("第"+str(k)+"支股票")
    tsla_data =final_data_real[k]

    tsla_data.columns
    tsla_close = tsla_data['close'].values
    #tsla_data=tsla_data.drop('Adj Close', axis=1)
   
    #train 60% val 20% test 20%   
    n = 50
    train =tsla_data[:int(len(tsla_data) *0.6)]
    val =tsla_data[int(len(tsla_data) *0.6):int(len(tsla_data) *0.8)]
    test =tsla_data[int(len(tsla_data) *0.8):]
    y_testc=test['close'][n:]
    #feature_names = list(train.drop('close', axis=1).columns)
    feature_names = list(train.columns)
    x_train = []
    y_train = []
    train_indexes = []
    norm_data_xtrain = train[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(train)-n)):#      
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainaddscalar=x_scaler.fit_transform(x_trainadd)

        x_train.append(np.transpose(x_trainaddscalar)) 
        y_train.append(train['close'].iloc[i+n]) 
        train_indexes.append(train.index[i+n]) 
    print(x_train[0])
    
    x_train=np.array(x_train)
    y_train_dataframe=pd.DataFrame(y_train).iloc[:len(y_train)]
    y_train_tran=y_scaler.fit_transform(y_train_dataframe)
    y_train=np.array(y_train_tran).reshape(-1)
    print(x_train.shape)
      
    x_val = []
    y_val = []
    val_indexes = []
    norm_data_xval = val[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(val)-n)):       
        x_valadd=norm_data_xval.iloc[i:i+n]. values
        x_valaddscalar=x_scaler.fit_transform(x_valadd)
        x_val.append(np.transpose(x_valaddscalar))
        y_val.append(val['close'].iloc[i+n]) 

    print(x_val[0])
    x_val=np.array(x_val)
    y_val_dataframe=pd.DataFrame(y_val).iloc[:len(y_train)]
    y_val_val=y_scaler.fit_transform(y_val_dataframe)
    y_val=np.array(y_val_val).reshape(-1)
    print(x_val.shape)    
      
    #test
    x_test = []
    y_test = []
    test_indexes = []
    
    norm_data_xtest = test[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(test)-n)): 
        x_testadd=norm_data_xtest.iloc[i:i+n]. values
        x_testaddscalar=x_scaler.fit_transform(x_testadd)
        x_test.append(np.transpose(x_testaddscalar))
        y_test.append(test['close'].iloc[i+n]) 
        test_indexes.append(test.index[i+n]) 



    x_test=np.array(x_test)

    y_test_dataframe=pd.DataFrame(y_test).iloc[:len(y_test)]
    y_test_tran=y_scaler.fit_transform(y_test_dataframe)
    y_test=np.array(y_test_tran).reshape(-1) 
    
    #timestep 10
    n = 50
    n_steps = n 
    n_features = 5
    model = keras.models.Sequential()
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_features,n_steps)))
    model.add(LSTM(20,activation='relu', input_shape = (n_features,n_steps)))
    model.add(Dense(1))
    model.summary()
    
    model.compile(keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()  loss=custom_mean_squared_error
    metrics=[keras.metrics.MeanAbsoluteError()])
    
    model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock/model8/'

    log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock', 'model8')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                                 monitor='val_mean_absolute_error', 
                                                 save_best_only=True, 
                                                 mode='min')
    history = model.fit(x_train,y_train,batch_size=32,epochs=30)
    
    #predict
    predictions = model.predict(x_test)
    predictions1=predictions.reshape(-1)
    predictions_orign = y_scaler.inverse_transform(predictions)
    

    meanmae_error=np.mean(abs(predictions_orign- np.array(y_testc).reshape(-1,1)))

    stock_mae.append(meanmae_error) 
    stock.append(stock_id[k]) 

#output
cnnlstm_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)
cnnlstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALAR CNN_LSTM1.csv', encoding='utf_8_sig')


