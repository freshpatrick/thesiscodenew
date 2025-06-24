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


# Load  dataset
output_directory = r'../../../../data/daily stock' 
output_path = os.path.join(output_directory, "IBM.csv")   
df=pd.read_csv(output_path)  
df=df.iloc[:,1:]
data_orign = df.drop(['Adj Close','Volume'], axis=1)
data=pd.concat([pd.DataFrame(df['Volume']),pd.DataFrame(data_orign)],axis=1)


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
time_step = 10
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
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='same',activation='sigmoid')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=3,padding='same',activation='sigmoid')(x)
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
        x = layers.Dense(10, activation="sigmoid")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(10, activation="sigmoid")(x)
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
    num_transformer_encoderblocks=8, 
    num_transformer_decoderblocks=range(0,2),
    mlp_dropout=0.2,
    dropout=0.2,
)


model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanAbsoluteError()])


model.summary()


#設定callback
model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'

log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')


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
plt.legend(loc='upper right')
