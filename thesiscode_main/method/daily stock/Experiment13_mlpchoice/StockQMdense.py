# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:30:11 2024

@author: 2507
"""

import csv
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



#load data
x_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\x_bigdata.npy')
y_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\y_bigdata.npy')


#random order
indexs=np.random.permutation(len(x_bigdata)) 
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]

#x
x_bigdata_array=np.array(x_bigdata)
X_train=x_bigdata_array[train_indexs]
X_val=x_bigdata_array[val_indexs]
X_test=x_bigdata_array[test_indexs]

#y
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]

y_train=y_scaler.fit_transform(pd.DataFrame(y_train))
y_val=y_bigdata_array[val_indexs]

y_val=y_scaler.fit_transform(pd.DataFrame(y_val))
y_test=y_bigdata_array[test_indexs]

y_test_orign=y_test
y_test=y_scaler.fit_transform(pd.DataFrame(y_test))





def build_model(
    input_shape,
    num_mlpblocks,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(inputs) 
    #mlpdense
    for dim in num_mlpblocks:
        x = layers.Dense(31, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x) 
    return keras.Model(inputs, outputs)
    
#Set hyperparameters
input_shape = X_train.shape[1:]
print(input_shape)



def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')

model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')

#跑幾層
mlp_mae=[]
for i in range(0,5):
    model = build_model(          
        input_shape,
        num_mlpblocks=range(0,i),  
        mlp_dropout=0.25)
        

    model.compile(keras.optimizers.Adam(0.001),
                  loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
                  metrics=[keras.metrics.MeanAbsoluteError()])


    model.summary()

    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5',                                                   
                                                    monitor='val_mean_absolute_error', 
                                                    save_best_only=True, 
                                                    mode='min')

    history = model.fit(X_train, y_train,                           
                        batch_size=32, 
                        epochs=30,  
                        validation_data=(X_val, y_val), 
                        callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


    #predict             
    test_predict = model.predict(X_test)
    test_predict = y_scaler.inverse_transform(test_predict)
    meanmae_error=np.mean(abs(test_predict- np.array(y_test_orign)))
    print(" 平均mae誤差: {:.2f}".format(meanmae_error)) 
    mlp_mae.append(meanmae_error)
        

    
finalmae=pd.DataFrame(mlp_mae)
finalmae.index=["1","2","3","4","5"]
finalmae.to_csv(r'D:/pytorch範例/transformer_tensorflow/1002transformer/mlp_dense/StockQMmlp.csv')














