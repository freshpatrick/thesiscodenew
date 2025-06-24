# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:45:22 2024

@author: User
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime

import numpy as np
import requests
import json
import time
import csv
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median#
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數

############################跑lstm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

###########神經元層
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention

# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm #使用進度條
import tensorflow as tf
from random import sample
import os

#load data
final_data_real=pd.read_csv(r'../../data/StockQM/405company.csv', encoding='utf_8_sig')


final_data_real=final_data_real.iloc[:,2:]
final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','return_rate','賣出價'], axis=1)
final_data_real.columns
final_data_real['長期資金佔不動產']=final_data_real['長期資金佔不動產']*1000
max(final_data_real['Net_Income'])
min(final_data_real['Net_Income'])
max(final_data_real['長期資金佔不動產'])
min(final_data_real['長期資金佔不動產'])
final_data_mean=final_data_real.iloc[:,2:]
x_scaler = MinMaxScaler(feature_range = (0, 1))
#使用x變數
x_variable=final_data_real.iloc[:,1:32]
x_variable=pd.DataFrame(x_scaler.fit_transform(x_variable))
#使用y變數
y_variable=pd.DataFrame(final_data_real.iloc[:,32])

company_name=pd.DataFrame(final_data_real.iloc[:,0])

#concat
columnname=final_data_real.columns
final_data_real=pd.concat([company_name,x_variable,y_variable],axis=1)
final_data_real.columns=columnname

print(final_data_real.isnull().sum())
isnull=final_data_real.isnull()
null_locat=np.where(isnull)

for j in range(0,len(null_locat[0])):
    null_row=null_locat[0][j]
    null_col=null_locat[1][j]
    nullbool=pd.isnull(final_data_real.iloc[null_row,null_col])

    if(nullbool==True):
        #找有值的最後一個數值取代
        print("***第"+str(j)+"***筆有缺值資料***")
        notnull=np.where(final_data_real.iloc[null_row,:].notnull())[0]
        final_data_real.iloc[null_row,null_col]=final_data_real.iloc[null_row,notnull[len(notnull)-1]]
        
print(final_data_real.isnull().sum())#無空值

x_bigdata = []
y_bigdata = []
yc_data=[]
stock_id=final_data_real['公司代號'].unique()
stock_mae=[] 
stock=[] 
final_data_real_copy=final_data_real

for k in range(0,len(stock_id)): 
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[final_data_real['公司代號']==stock_id[k]]    
    
    if(stock_id[k]==4142): 
        continue
    final_data=final_data.drop('公司代號', axis=1)
    n = 2 
    feature_names = list(final_data.drop('新賣出價', axis=1).columns)

    train_indexes = []
    norm_data_xtrain = final_data[feature_names]    
    norm_data_yctrain = pd.DataFrame(final_data['新賣出價'])
        

    for i in tqdm.tqdm_notebook(range(0,len(final_data)-n)):   
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainadd=np.transpose(x_trainadd)
        x_bigdata.append( x_trainadd)
        yc_trainadd=norm_data_yctrain[i:i+n]
        yc_data.append(yc_trainadd)     
        y_bigdata.append(final_data['新賣出價'].iloc[i+n-1]) 
    print(x_bigdata[0])

    
    
np.save(r'x_bigdata', np.array(x_bigdata))
np.save(r'y_bigdata', np.array(y_bigdata))


##noadd10days
final_data_real=pd.read_csv(r'../../data/StockQM/405company.csv', encoding='utf_8_sig')
final_data_real=final_data_real.iloc[:,2:]
final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','return_rate','賣出價' ,'前10天', '前9天', '前8天', '前7天', '前6天', '前5天',
'前4天', '前3天', '前2天', '前1天'], axis=1)
final_data_real.columns
final_data_real['長期資金佔不動產']=final_data_real['長期資金佔不動產']*1000
max(final_data_real['Net_Income'])
min(final_data_real['Net_Income'])
max(final_data_real['長期資金佔不動產'])
min(final_data_real['長期資金佔不動產'])
final_data_mean=final_data_real.iloc[:,2:]
x_scaler = MinMaxScaler(feature_range = (0, 1))
#使用x變數
x_variable=final_data_real.iloc[:,1:22]
x_variable=pd.DataFrame(x_scaler.fit_transform(x_variable))
#使用y變數
y_variable=pd.DataFrame(final_data_real.iloc[:,22])

company_name=pd.DataFrame(final_data_real.iloc[:,0])

#concat
columnname=final_data_real.columns
final_data_real=pd.concat([company_name,x_variable,y_variable],axis=1)
final_data_real.columns=columnname

print(final_data_real.isnull().sum())
isnull=final_data_real.isnull()
null_locat=np.where(isnull)

for j in range(0,len(null_locat[0])):
    null_row=null_locat[0][j]
    null_col=null_locat[1][j]
    nullbool=pd.isnull(final_data_real.iloc[null_row,null_col])

    if(nullbool==True):
        #找有值的最後一個數值取代
        print("***第"+str(j)+"***筆有缺值資料***")
        notnull=np.where(final_data_real.iloc[null_row,:].notnull())[0]
        final_data_real.iloc[null_row,null_col]=final_data_real.iloc[null_row,notnull[len(notnull)-1]]
        
print(final_data_real.isnull().sum())#無空值

x_bigdata = []
y_bigdata = []
yc_data=[]
stock_id=final_data_real['公司代號'].unique()
stock_mae=[] 
stock=[] 
final_data_real_copy=final_data_real

for k in range(0,len(stock_id)): 
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[final_data_real['公司代號']==stock_id[k]]    
    
    if(stock_id[k]==4142): 
        continue
    final_data=final_data.drop('公司代號', axis=1)
    n = 2 
    feature_names = list(final_data.drop('新賣出價', axis=1).columns)

    train_indexes = []
    norm_data_xtrain = final_data[feature_names]    
    norm_data_yctrain = pd.DataFrame(final_data['新賣出價'])
        

    for i in tqdm.tqdm_notebook(range(0,len(final_data)-n)):   
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainadd=np.transpose(x_trainadd)
        x_bigdata.append( x_trainadd)
        yc_trainadd=norm_data_yctrain[i:i+n]
        yc_data.append(yc_trainadd)     
        y_bigdata.append(final_data['新賣出價'].iloc[i+n-1]) 
    print(x_bigdata[0])    


np.save(r'x_bigdata_noadd10traddays', np.array(x_bigdata))
np.save(r'y_bigdata_noadd10traddays', np.array(y_bigdata))