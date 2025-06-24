# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:45:42 2024

@author: User
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


# Load  dataset
output_directory = r'C:\Users\2507\Desktop\遠端資料\data\15mindata\MSFT'
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
y_scaler = MinMaxScaler(feature_range=(0, 1))


x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
data_scaled=np.array(data_scaled)

def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :5]  
        dataX.append(a)
        dataY.append(dataset[i + time_step, 4]) 
    return np.array(dataX), np.array(dataY)



# Parameters
time_step = 50
training_size = int(len(data_scaled) * 0.6)
validat_size=int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - validat_size
train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]



X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs 
    #Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #Branch
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='same',activation='relu')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=3,padding='same',activation='relu')(x)
    x = layers.Concatenate()([x_1, x_2])
       
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=5, kernel_size=2,padding='same')(x)
    return x + res




def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_encoderblocks,
    num_transformer_decoderblocks,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    #encoder
    for _ in range(num_transformer_encoderblocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) 
    x_encoder1=x 
    #decoder
    for dim in num_transformer_decoderblocks:
        x_encoder=x
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x=x_encoder+x
    x=layers.Concatenate()([x_encoder1, x])
    outputs = layers.Dense(1)(x) 
    return keras.Model(inputs, outputs)
    


#Set hyperparameters
input_shape = X_train.shape[1:]
print(input_shape)
epoch_number=30



model = build_model(
    input_shape,
    head_size=64,
    num_heads=2,  
    ff_dim=4,
    num_transformer_encoderblocks=2, 
    num_transformer_decoderblocks=range(0,2), 
    dropout=0.25,
)


model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              metrics=[keras.metrics.MeanAbsoluteError()])


model.summary()


def get_flops(model):
  tf.compat.v1.disable_eager_execution()  #關閉eager狀態
  sess = tf.compat.v1.Session()#自動轉換腳本

  run_meta = tf.compat.v1.RunMetadata()
  profiler = tf.compat.v1.profiler
  #opts = tf.profiler.ProfileOptionBuilder.float_operation()
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  # We use the Keras session graph in the call to the profiler.
  flops = profiler.profile(graph=sess.graph, 
                           run_meta=run_meta, cmd='op', options=opts)

  return flops.total_float_ops  # Prints the "flops" of the model


get_flops(model) 