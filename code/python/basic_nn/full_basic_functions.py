import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values).float()
        self.y = torch.from_numpy(y).float()
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            ) 
    def forward(self, x):
        #x = self.flatten(x)
        return self.linear_relu_stack(x)

def process_data(cases, test_size):


    cases_train = cases[cases['time'] <= cases['time'].quantile(1 - test_size)]
    cases_test = cases[cases['time'] > cases['time'].quantile(1 - test_size)]

    id_train = cases_train[['time', 'city']]
    id_test = cases_test[['time', 'city']]

    y_train = cases_train['cases'].to_numpy()
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = cases_test['cases'].to_numpy()
    y_test = y_test.reshape((y_test.shape[0], 1))

    #X_train = train_df[abl_cols]
    X_train = cases_train
    X_train = X_train.drop(['time', 'city', 'cases', 'susc', 'pop'], axis = 1)
    X_train = X_train.drop(['births'], axis = 1)

    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nbc')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nearest_big_city')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='susc')))]
#    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nc')))]

    #X_test = test_df[abl_cols]
    X_test = cases_test
    X_test = X_test.drop(['time', 'city', 'cases', 'susc', 'pop'], axis = 1)
    X_test = X_test.drop(['births'], axis = 1)

    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nbc')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nearest_big_city')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='susc')))]
#    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nc')))]


    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)


    return train_data, test_data, X_train.shape[1], id_train, id_test

def get_dataloaders(train_data, test_data, batch_size):
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, pin_memory = True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory = True)

    return train_dataloader, test_dataloader



def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for dataloader_iter, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if dataloader_iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, dataloader_iter * len(X), len(train_loader.dataset),
                100. * dataloader_iter / len(train_loader), loss.item()))
            if args.dry_run:
                break       

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()  

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss, len(test_loader.dataset) * test_loader.batch_size))
