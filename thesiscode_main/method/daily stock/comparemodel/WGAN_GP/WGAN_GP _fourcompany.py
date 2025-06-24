# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:58:18 2024

@author: 2507
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import  os
import yfinance as yf
from sklearn.metrics import mean_absolute_error

#load data
AAPL= yf.download("AAPL", start="1980-01-01", end="2024-07-31")
TSLA= yf.download("TSLA", start="1980-01-01", end="2024-07-31")
MSFT= yf.download("MSFT", start="1980-01-01", end="2024-07-31")
IBM = yf.download("IBM ", start="1980-01-01", end="2024-07-31")


final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)


stock_id=['AAPL','TSLA','MSFT','IBM']

stock_mae=[] 
stock=[] 


class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar



class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out_6 = self.linear_3(out_5)
        return out_6

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size = 5, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 5, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 5, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.linear2 = nn.Linear(220, 220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x =  conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out = self.linear3(out_2)
        return out




def sliding_window(x, y, window):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan



final_data_real_copy=final_data_real

#parameter
num_epochs = 10
learning_rate = 0.00003
batch_size = 32
learning_rate = 0.000115


for k in range(0,len(stock_id)):  #len(stock_id)
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[k]

    data=final_data
    data.drop('Adj Close', axis=1)



    n = 10 
    train =data[:int(len(data) *0.8)]
    test =data[int(len(data) *0.8):]
    y_testc=test['Close'][n:]
    train_x = np.array(train.drop('Close', axis=1))
    train_y = np.array(train['Close'])


    test_x=np.array(test.drop('Close', axis=1))
    test_y = np.array(test['Close'])
    x_scaler = MinMaxScaler(feature_range = (0, 1))
    y_scaler = MinMaxScaler(feature_range = (0, 1))
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)
    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), batch_size = 10, shuffle = False)
    model = VAE([5, 400, 400, 400, 10], 10)  
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        total_loss = 0
        loss_ = []
        for (x, ) in train_loader:
            x = x.to(device)
            output, z, mu, logVar = model(x)
            kl_divergence = 0.5* torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(output, x) + kl_divergence
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        hist[epoch] = sum(loss_)
        print('[{}/{}] Loss:'.format(epoch+1, num_epochs), sum(loss_))
    ##模型套入VAE##
    model.eval()
    _, VAE_train_x, train_x_mu, train_x_var = model(torch.from_numpy(train_x).float().to(device))
    _, VAE_test_x, test_x_mu, test_x_var = model(torch.from_numpy(test_x).float().to(device))

    ##sliding window
    train_x = np.concatenate((train_x, VAE_train_x.cpu().detach().numpy()), axis = 1)
    test_x = np.concatenate((test_x, VAE_test_x.cpu().detach().numpy()), axis = 1)
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    learning_rate = 0.000115
    critic_iterations = 5
    weight_clip = 0.01
    train_x_slide, train_y_slide, train_y_gan = sliding_window(train_x, train_y, 3)
    test_x_slide, test_y_slide, test_y_gan = sliding_window(test_x, test_y, 3)
    print(f'train_x: {train_x_slide.shape} train_y: {train_y_slide.shape} train_y_gan: {train_y_gan.shape}')
    print(f'test_x: {test_x_slide.shape} test_y: {test_y_slide.shape} test_y_gan: {test_y_gan.shape}')


    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size = batch_size, shuffle = False)

    modelG = Generator(15).to(device)  ##我有修改
    modelD = Discriminator().to(device)

    optimizerG = torch.optim.Adam(modelG.parameters(), lr = learning_rate, betas = (0.0, 0.9), weight_decay = 1e-3)
    optimizerD = torch.optim.Adam(modelD.parameters(), lr = learning_rate, betas = (0.0, 0.9), weight_decay = 1e-3)

    histG = np.zeros(num_epochs)
    histD = np.zeros(num_epochs)
    count = 0

    print('資料型態為'+str(type(trainDataloader)))

    for epoch in range(num_epochs):
        loss_G = []
        loss_D = []
        for (x, y) in trainDataloader:
            x = x.to(device)
            y = y.to(device)
            fake_data = modelG(x)
            fake_data = torch.cat([y[:, :3, :], fake_data.reshape(-1, 1, 1)], axis = 1)
            critic_real = modelD(y)
            critic_fake = modelD(fake_data)
            lossD = -(torch.mean(critic_real) - torch.mean(critic_fake))
            modelD.zero_grad()
            lossD.backward(retain_graph = True)
            optimizerD.step()

            output_fake = modelD(fake_data)
            lossG = -torch.mean(output_fake)
            modelG.zero_grad()
            lossG.backward()
            optimizerG.step()

            loss_D.append(lossD.item())
            loss_G.append(lossG.item())
        histG[epoch] = sum(loss_G)
        histD[epoch] = sum(loss_D)
        print(f'[{epoch+1}/{num_epochs}] LossD: {sum(loss_D)} LossG:{sum(loss_G)}')

    modelG.eval()
    pred_y_train = modelG(train_x_slide.to(device))
    pred_y_test = modelG(test_x_slide.to(device))

    y_train_true = y_scaler.inverse_transform(train_y_slide)
    y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

    y_test_true = y_scaler.inverse_transform(test_y_slide)
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    MAE = mean_absolute_error(y_test_true, y_test_pred)
    meanmae_error=np.mean(MAE)

    stock_mae.append(meanmae_error) 
    stock.append(stock_id[k]) 

    print(" 平均mae誤差: {:.2f}".format(meanmae_error))

    print("所有股票平均mae誤差: {:.2f}".format(meanmae_error))
    print("******************第"+str(k)+"支股票***************")


wgan_gpdata=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)
    ###############20240507 自己修結束###########
wgan_gpdata.to_csv(r'D:/pytorch範例/wgangp範例/WGAN_GP stock.csv', encoding='utf_8_sig')
