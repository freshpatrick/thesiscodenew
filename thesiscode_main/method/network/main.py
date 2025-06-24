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
import AAPL_network
from AAPL_network import *
import TSLA_network
from TSLA_network import *
import IBM_network
from IBM_network import *
import MSFT_network
from MSFT_network import *
import AAPL_15minnetwork
from AAPL_15minnetwork import *
import TSLA_15minnetwork
from TSLA_15minnetwork import *
import IBM_15minnetwork
from IBM_15minnetwork import *
import MSFT_15minnetwork
from MSFT_15minnetwork import *
import AStock_network
from AStock_network import *
import StockQM_network
from StockQM_network import *




#function
def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :5] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 4]) 
    return np.array(dataX), np.array(dataY)

def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')


x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))


#class
# Load  dataset
stocklist=["AAPL","MSFT","TSLA","IBM","AAPL15min","MSFT15min","TSLA15min","IBM15min","StockQM","Astock"]
for i in range(6,7):  #len(stocklist)  
    stockname=stocklist[i]
 
    if (stockname=="AAPL"):
       # Load  dataset
       output_directory = r'../../data/daily stock' 
       output_path = os.path.join(output_directory, "AAPL.csv")   
       df=pd.read_csv(output_path)  
       df=df.iloc[:,1:]   
       data_orign = df.drop(['Adj Close', 'Volume'], axis=1)
       data=pd.concat([pd.DataFrame(df['Volume']),pd.DataFrame(data_orign)],axis=1)      
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)

       # Parameters
       time_step = 10
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30



       #model
       Stockmodel = AAPL_network.StockAAPLModel()      
       #Stockmodel = AAPL_network.StockAAPLModel    
       #Stockmodel.input_shape = X_train.shape[1:]       
       #from AAPL_network import StockAAPLModel  # 確保匯入正確
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback       
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/AAPL.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


       model.load_weights(model_dir + '/AAPL.h5')
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


    if (stockname=="MSFT"):
       # Load  dataset
       output_directory = r'../../data/daily stock' 
       output_path = os.path.join(output_directory, "MSFT.csv")   
       df=pd.read_csv(output_path)  
       df=df.iloc[:,1:]
       
       data_orign = df.drop(['Adj Close', 'Volume'], axis=1)
       data=pd.concat([pd.DataFrame(df['Volume']),pd.DataFrame(data_orign)],axis=1)
       
       
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)

       # Parameters
       time_step = 10
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30



       #model
       Stockmodel = StockMSFTModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/MSFT1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])

       model.load_weights(model_dir + '/MSFT1.h5')
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

    if (stockname=="TSLA"):
       # Load  dataset
       output_directory = r'../../data/daily stock' 
       output_path = os.path.join(output_directory, "TSLA.csv")   
       df=pd.read_csv(output_path)  
       df=df.iloc[:,1:]
       
       data_orign = df.drop(['Adj Close', 'Volume'], axis=1)
       data=pd.concat([pd.DataFrame(df['Volume']),pd.DataFrame(data_orign)],axis=1)
       
       
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)

       # Parameters
       time_step = 10
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30



       #model
       Stockmodel = StockTSLAModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/TSLA.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


       # Make predictions
       model.load_weights(model_dir + '/TSLA1.h5')
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

    if (stockname=="IBM"):
       # Load  dataset
       output_directory = r'../../data/daily stock' 
       output_path = os.path.join(output_directory, "MSFT.csv")   
       df=pd.read_csv(output_path)  
       df=df.iloc[:,1:]
       
       data_orign = df.drop(['Adj Close', 'Volume'], axis=1)
       data=pd.concat([pd.DataFrame(df['Volume']),pd.DataFrame(data_orign)],axis=1)
       
       
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)

       # Parameters
       time_step = 10
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30



       #model
       Stockmodel = StockIBMModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/IBM15min1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=5, 
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

    if (stockname=="AAPL15min"):
       # Load  dataset
       output_directory = r'../../data/15 minutes stock' 
       output_path = os.path.join(output_directory, "AAPL15min.csv")   
       df=pd.read_csv(output_path)  
       data=df.iloc[:,2:]
       vol=data['volume']
       close=data['close']
       #volume和close欄位對調
       data['close']=vol
       data['volume']=close
       data.columns=['open', 'high', 'low', 'volume', 'close']
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)  
       # Parameters
       time_step = 50
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30

       #model
       Stockmodel = StockAAPL15minModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/AAPL15min.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


       # Make predictions
       model.load_weights(model_dir + '/AAPL15minute1.h5')
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
       
    if (stockname=="MSFT15min"):
       # Load  dataset
       output_directory = r'../../data/15 minutes stock' 
       output_path = os.path.join(output_directory, "MSFT15min.csv")   
       df=pd.read_csv(output_path)  
       data=df.iloc[:,2:]
       vol=data['volume']
       close=data['close']
       #volume和close欄位對調
       data['close']=vol
       data['volume']=close
       data.columns=['open', 'high', 'low', 'volume', 'close']
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)
       # Parameters
       time_step = 50
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30

       #model
       Stockmodel = StockMSFT15minModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/MSFT15min.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


       # Make predictions
       model.load_weights(model_dir + '/MSFT15minute1.h5')
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

    if (stockname=="TSLA15min"):
       # Load  dataset
       output_directory = r'../../data/15 minutes stock' 
       output_path = os.path.join(output_directory, "TSLA15min.csv")   
       df=pd.read_csv(output_path)  
       data=df.iloc[:,2:]
       vol=data['volume']
       close=data['close']
       #volume和close欄位對調
       data['close']=vol
       data['volume']=close
       data.columns=['open', 'high', 'low', 'volume', 'close']
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)
       # Parameters
       time_step = 50
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30

       #model
       Stockmodel = StockTLSA15minModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/TSLA15min.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



       history = model.fit(X_train, y_train, 
                batch_size=32,
                epochs=30, 
                validation_data=(X_val, y_val),  
                callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


       # Make predictions
       model.load_weights(model_dir + '/TSLA15minute1.h5')
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

    if (stockname=="IBM15min"):
       # Load  dataset
       output_directory = r'../../data/15 minutes stock' 
       output_path = os.path.join(output_directory, "IBM15min.csv")   
       df=pd.read_csv(output_path)  
       data=df.iloc[:,2:]
       vol=data['volume']
       close=data['close']
       #volume和close欄位對調
       data['close']=vol
       data['volume']=close
       data.columns=['open', 'high', 'low', 'volume', 'close']
       x_data_scaled = x_scaler.fit_transform(data.iloc[:,0:5])  
       y_data_scaled = y_scaler.fit_transform(np.array(data)[:,4:5])  
       data_scaled = pd.concat([pd.DataFrame(x_data_scaled),pd.DataFrame(y_data_scaled)],axis=1)
       data_scaled=np.array(data_scaled)
       # Parameters
       time_step = 50
       training_size = int(len(data_scaled) * 0.6)
       validat_size=int(len(data_scaled) * 0.8)
       test_size = len(data_scaled) - validat_size
       train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]

       X_train, y_train = create_dataset(train_data, time_step)
       X_val, y_val = create_dataset(val_data, time_step)
       X_test, y_test = create_dataset(test_data, time_step)

       #Set hyperparameters
       input_shape = X_train.shape[1:]
       print(input_shape)
       epoch_number=30

       #model
       Stockmodel = StockIBM15minModel()
       model=Stockmodel.callmodel()
       
       model.compile(keras.optimizers.Adam(0.001),
                     
                     loss=keras.losses.MeanSquaredError(),
                     metrics=[keras.metrics.MeanAbsoluteError()])


       model.summary()


       #設定callback
       model_dir = r'../../checkpoint'
       log_dir = os.path.join(r'../../checkpoint', 'model')
       model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
       model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/IBM15min.h5', 
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
    if (stockname=="StockQM"):
       # Load  dataset
      x_bigdata=np.load(r'../../data/StockQM/x_bigdata.npy')
      y_bigdata=np.load(r'../../data/StockQM/y_bigdata.npy')
      indexs=np.random.permutation(len(x_bigdata)) 
      train_indexs=indexs[:int(len(x_bigdata)*0.6)]
      val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
      test_indexs=indexs[int(len(x_bigdata)*0.8):]

      #x
      x_bigdata_array=np.array(x_bigdata)
      x_train=x_bigdata_array[train_indexs]
      x_val=x_bigdata_array[val_indexs]
      x_test=x_bigdata_array[test_indexs]

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


      #Set hyperparameters
      input_shape = x_train.shape[1:]
      print(input_shape)
      epoch_number=30
      batch_size=64



      #model
      Stockmodel = StockQMModel()
      model=Stockmodel.callmodel()


      model.compile(keras.optimizers.Adam(0.001),
                    loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
                    metrics=[keras.metrics.MeanAbsoluteError()])


      model.summary()


      #callback
      model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'
      log_dir = os.path.join(r'../../checkpoint', 'model')
      model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
      model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/StockQM.h5', 
                                              monitor='val_mean_absolute_error', 
                                              save_best_only=True, 
                                              mode='min')


      history = model.fit(x_train, y_train,  
                     batch_size=32,  
                     epochs=30,  
                     validation_data=(x_val, y_val),  
                     callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])


      history.history.keys() 



      y_pred = model.predict(x_test)
      y_pred = y_scaler.inverse_transform(y_pred)
      meanmae_error=np.mean(abs(y_pred- np.array(y_test_orign)))
      print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    if (stockname=="Astock"):
      # Load  dataset
      x_train=np.load(r'../../data/Astock/Astockx_train.npy')
      x_val=np.load(r'../../data/Astock/Astockx_val.npy')
      x_test=np.load(r'../../data/Astock/Astockx_test.npy')
      y_train=np.load(r'../../data/Astock/Astocky_train.npy')
      y_val=np.load(r'../../data/Astock/Astocky_val.npy')
      y_test=np.load(r'../../data/Astock/Astocky_test.npy')

      y_test_orign=y_test
      y_scaler = MinMaxScaler(feature_range = (0, 1))
      y_train=tf.one_hot(y_train,3)
      y_val=tf.one_hot(y_val,3)
      y_test=tf.one_hot(y_test,3)
      #Set hyperparameters
      input_shape = x_train.shape[1:]
      print(input_shape)
      epoch_number=30
      batch_size=64
      #model
      Stockmodel = AStockodel()
      model=Stockmodel.callmodel()


      model.compile(keras.optimizers.Adam(0.001),
                    loss=keras.losses.CategoricalCrossentropy(),  #loss=keras.losses.MeanSquaredError()
                    metrics=[keras.metrics.CategoricalAccuracy()])


      model.summary()


      #callback
      model_dir = r'../../checkpoint'
      log_dir = os.path.join(r'../../checkpoint', 'model')
      model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
      model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Astock.h5', 
                                              monitor='val_categorical_accuracy', 
                                              save_best_only=True, 
                                              mode='max')



      history = model.fit(x_train, y_train,  
                     batch_size=32,  
                     epochs=30,  
                     validation_data=(x_val, y_val),  
                     callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])




      history.history.keys() 



      y_pred = model.predict(x_test)



      #accuracy
      y_prelabel=[]
      for j in range(0,len(y_pred)):
          y_label=np.where(y_pred[j] ==max(y_pred[j])) 
          y_prelabel.append(y_label[0][0])
          
      y_prelabel=np.array(y_prelabel)    
      accuracy=(y_prelabel==y_test_orign).mean()
      print("準確率為"+str(accuracy*100)+"%")