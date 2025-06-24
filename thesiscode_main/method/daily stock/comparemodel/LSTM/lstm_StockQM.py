# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:05:40 2024

@author: User
"""
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout,Conv2D



#load data
x_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\x_bigdata.npy')
y_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\y_bigdata.npy')


indexs=np.random.permutation(len(x_bigdata)) 
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]


x_bigdata_array=np.array(x_bigdata)
x_train=x_bigdata_array[train_indexs]
x_val=x_bigdata_array[val_indexs]
x_test=x_bigdata_array[test_indexs]

y_scaler = MinMaxScaler(feature_range = (0, 1))
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]
y_train=y_scaler.fit_transform(pd.DataFrame(y_train))
y_val=y_bigdata_array[val_indexs]
y_val=y_scaler.fit_transform(pd.DataFrame(y_val))
y_test=y_bigdata_array[test_indexs]
y_test_orign=y_test
y_test=y_scaler.fit_transform(pd.DataFrame(y_test))


n = 2
n_steps = n 
n_features = 31
model = keras.models.Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_features,n_steps)))
model.add(LSTM(50,activation='relu'))
model.add(Dense(1))
model.summary()
    


model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanAbsoluteError(),
              metrics=[keras.metrics.MeanAbsoluteError()])



model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/model9/'
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model9')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')


history = model.fit(x_train, y_train, 
               batch_size=32, 
               epochs=30, 
               validation_data=(x_val, y_val),  
               callbacks=[model_cbk, model_mckp]) 



y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    
    
    
    