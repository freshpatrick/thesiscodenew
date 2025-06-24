# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:08:17 2024

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf



AAPL= yf.download("AAPL", start="1980-01-01", end="2024-07-31")
TSLA= yf.download("TSLA", start="1980-01-01", end="2024-07-31")
MSFT= yf.download("MSFT", start="1980-01-01", end="2024-07-31")
IBM= yf.download("IBM", start="1980-01-01", end="2024-07-31")



#AAPL= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AAPL.csv', encoding='utf_8_sig')
#AMZN= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AMZN.csv', encoding='utf_8_sig')
#GOOG= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/GOOG.csv', encoding='utf_8_sig')
#MSFT= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/MSFT.csv', encoding='utf_8_sig')
#TSLA= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/TSLA.csv', encoding='utf_8_sig')

final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)

stock_id=['AAPL','TSLA','MSFT','IBM']
stock_mae=[] 
stock=[] 

x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))
for k in range(0,5):  #
    print("第"+str(k)+"支股票")
    df = final_data_real[k]
    df['Date']=df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].apply(lambda date: date.toordinal())
    X = df.drop(["Adj Close","Date"], axis=1)
    y = df["Adj Close"]
    train_y =y[:int(len(y) *0.8)]
    price_mean=y.mean() 
    price_std=y.std()  
    y_testc=np.array(df["Adj Close"][int(len(df) *0.8):])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train=x_scaler.fit_transform(X_train)
    y_train=y_scaler.fit_transform(pd.DataFrame(y_train)).reshape(-1)  
    X_test=x_scaler.fit_transform(X_test)
    y_train=y_scaler.fit_transform(pd.DataFrame(y_train)).reshape(-1)
    y_test=y_scaler.fit_transform(pd.DataFrame(y_test)).reshape(-1)
    rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    y_pred = y_scaler.inverse_transform(pd.DataFrame(y_pred))
    y_pred=np.reshape(y_pred,y_test.shape)
    mae = np.mean(np.abs(y_testc - y_pred))
    stock_mae.append(mae) #股票MSE
    stock.append(stock_id[k]) #股票名稱


big_rfr_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)

big_rfr_data.mean() #46.110392

big_rfr_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALARfive_outputdata_randomforest_regresion.csv', encoding='utf_8_sig')


    