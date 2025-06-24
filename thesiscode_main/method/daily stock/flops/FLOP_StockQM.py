# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:30:11 2024

@author: 2507
"""

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


##設定步驟##
indexs=np.random.permutation(len(x_bigdata)) #隨機排序
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]

#x部分
x_bigdata_array=np.array(x_bigdata)
x_train=x_bigdata_array[train_indexs]
x_val=x_bigdata_array[val_indexs]
x_test=x_bigdata_array[test_indexs]

#y部分
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]
#新
y_train=y_scaler.fit_transform(pd.DataFrame(y_train))
y_val=y_bigdata_array[val_indexs]
#新
y_val=y_scaler.fit_transform(pd.DataFrame(y_val))
y_test=y_bigdata_array[test_indexs]
#新
y_test_orign=y_test
y_test=y_scaler.fit_transform(pd.DataFrame(y_test))


##開始建構transformer模型
#Build the model
from tensorflow import keras
from tensorflow.keras import layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # Normalization    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #Branch
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='relu')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=4,activation='relu')(x)
    x = layers.Concatenate()([x_1, x_2])
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res




def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_encoderblocks,
    num_transformer_decoderblocks,
    #mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_encoderblocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) 
    x_encoder1=x 
    #decoder
    for dim in num_transformer_decoderblocks:
        x_encoder=x
        x = layers.Dense(31, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(31, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x=x_encoder+x
    x=layers.Concatenate()([x_encoder1, x])
    outputs = layers.Dense(1, activation="relu")(x) 
    return keras.Model(inputs, outputs)
    


#Set hyperparameters
input_shape = x_train.shape[1:]
print(input_shape)
epoch_number=30
batch_size=64



model = build_model(
    input_shape,
    head_size=64,
    num_heads=2,  
    ff_dim=4,
    num_transformer_encoderblocks=8, 
    num_transformer_decoderblocks=range(0,6), 
    mlp_dropout=0.25,
    dropout=0.25,
)


model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              metrics=[keras.metrics.MeanAbsoluteError()])


model.summary()


def get_flops(model):
  tf.compat.v1.disable_eager_execution()  
  sess = tf.compat.v1.Session()

  run_meta = tf.compat.v1.RunMetadata()
  profiler = tf.compat.v1.profiler
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  flops = profiler.profile(graph=sess.graph, 
                           run_meta=run_meta, cmd='op', options=opts)
  return flops.total_float_ops  


get_flops(model)  