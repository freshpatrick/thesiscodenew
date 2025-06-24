# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:20:19 2024

@author: 2507
"""
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import os
import keras
from keras.layers import Flatten
import yfinance as yf
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
from tensorflow.python.framework import ops
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load  dataset
output_directory =  r'C:\Users\2507\Desktop\遠端資料\data\15mindata\MSFT'
output_path = os.path.join(output_directory, "MSFT15min.csv")   
df=pd.read_csv(output_path)  
data=df.iloc[:,2:]

vol=data['volume']
close=data['close']
#volume和close欄位對調
data['close']=vol
data['volume']=close
data.columns=['open', 'high', 'low', 'volume', 'close']

x_scaler = MinMaxScaler(feature_range=(0, 1))

x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
y_bigdata = np.array(data)[:,4:5]  

###############開始跑資料################
X = data.drop(["close"], axis=1)
y = df["close"]

train_y =y[:int(len(y) *0.8)]
price_mean=y.mean() 
price_std=y.std()  
y_testc=np.array(df["close"][int(len(df) *0.8):])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train=x_scaler.fit_transform(X_train)
#y_train=y_scaler.fit_transform(pd.DataFrame(y_train)).reshape(-1)  
X_test=x_scaler.fit_transform(X_test)
#y_train=pd.DataFrame(y_train).reshape(-1)
#y_test=pd.DataFrame(y_test).reshape(-1)
rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
#y_pred = y_scaler.inverse_transform(pd.DataFrame(y_pred))
y_pred=np.reshape(y_pred,y_test.shape)
mae = np.mean(np.abs(y_testc - y_pred))
print(" 平均mae誤差: {:.2f}".format(mae))






















##跑模型## 80%訓練  20%測試
indexs=np.random.permutation(len(x_data_scaled)) #隨機排序 49005以下的數字
train_indexs=indexs[:int(len(x_data_scaled)*0.6)]
test_indexs=indexs[int(len(x_data_scaled)*0.6):]
#x部分
x_train=np.array(x_data_scaled.loc[x_data_scaled.index.intersection(train_indexs),:])  
x_test=np.array(x_data_scaled.loc[x_data_scaled.index.intersection(test_indexs),:])  
#y部分
y_bigdata=pd.DataFrame(y_bigdata)
y_train=np.array(y_bigdata.loc[y_bigdata.index.intersection(train_indexs),:]).reshape(-1)
y_test=np.array(y_bigdata.loc[y_bigdata.index.intersection(test_indexs),:]).reshape(-1)
# RandomFores 迴歸
rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
rfr.fit(x_train, y_train)
# Predict 預測y
y_pred = rfr.predict(x_test)
y_pred=np.reshape(y_pred,y_test.shape)
# 計算MAE
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
#meanmae_error   0.044
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
