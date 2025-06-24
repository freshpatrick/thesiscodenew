# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:36:51 2024

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
output_directory = r'C:\Users\2507\Desktop\遠端資料\data\15mindata\IBM'
output_path = os.path.join(output_directory, "IBM15min.csv")   
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
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


#建立模型顯示出要用幾次
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
    for _ in range(num_transformer_encoderblocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #decoder
    for dim in num_transformer_decoderblocks:
        x_conv=layers.Conv1D(filters=4, kernel_size=1, activation='relu',padding='same')(x)
        x =layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=1,activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout)(x)
        x =layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout)(x)
        x=x + x_conv
        
    x=layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = layers.Dense(1)(x)  
    return keras.Model(inputs, outputs)
    


#Set hyperparameters
input_shape = X_train.shape[1:]
print(input_shape)
epoch_number=30
batch_size=32



#make matrix
bigmae=np.zeros(shape=(10,10))


def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')

model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')


model = build_model(          
    input_shape,
    head_size=64, #256
    num_heads=2,  #4
    ff_dim=4,
    num_transformer_encoderblocks=6,  #IBM用2 validation會比較變動  4
    num_transformer_decoderblocks=range(0,1), #mlp_units=range(0,5),
    mlp_dropout=0.25,
    dropout=0.25)
        
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


# Make predictions
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)
# Inverse transform predictions
train_predict = y_scaler.inverse_transform(train_predict)
val_predict = y_scaler.inverse_transform(val_predict)
test_predict = y_scaler.inverse_transform(test_predict)


#output
y_testorign=y_scaler.inverse_transform(data_scaled[:,4:5])[(validat_size+time_step+1):len(data_scaled),:]
meanmae_error=np.mean(abs(test_predict- np.array(y_testorign)))
test_rmse = math.sqrt(mean_squared_error(test_predict, np.array(y_testorign)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error)) 
print(" RMSE誤差: {:.2f}".format(test_rmse))  


#plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('loss function')
plt.ylabel('loss')
plt.xlabel('epochs')
