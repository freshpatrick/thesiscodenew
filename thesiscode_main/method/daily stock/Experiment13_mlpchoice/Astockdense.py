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
x_train=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_train.npy')
x_val=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_val.npy')
x_test=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_test.npy')
y_train=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_train.npy')
y_val=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_val.npy')
y_test=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_test.npy')

y_test_orign=y_test
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_train=tf.one_hot(y_train,3)
y_val=tf.one_hot(y_val,3)
y_test=tf.one_hot(y_test,3)





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
        x = layers.Dense(25, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3,activation="softmax")(x) 
    return keras.Model(inputs, outputs)
    
#Set hyperparameters
input_shape = x_train.shape[1:]
print(input_shape)
epoch_number=30
batch_size=64



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
                                                    monitor='val_categorical_accuracy', 
                                                    save_best_only=True, 
                                                    mode='max')

    history = model.fit(x_train, y_train,                          
                        batch_size=32, 
                        epochs=30,  
                        validation_data=(x_val, y_val), 
                        callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])





    #predict                        
    y_pred = model.predict(x_test)
    #accuracy
    y_prelabel=[]
    for k in range(0,len(y_pred)):
        y_label=np.where(y_pred[k] ==max(y_pred[k])) 
        y_prelabel.append(y_label[0][0])
            
    y_prelabel=np.array(y_prelabel)    
    accuracy=(y_prelabel==y_test_orign).mean()
    accuracy=(y_prelabel==y_test_orign).mean()
    print("準確率為"+str(accuracy*100)+"%")
    mlp_mae.append(accuracy)
        

    
finalmae=pd.DataFrame(mlp_mae)
finalmae.index=["1","2","3","4","5"]
finalmae.to_csv(r'D:/pytorch範例/transformer_tensorflow/1002transformer/mlp_dense/Astockmlp.csv')














