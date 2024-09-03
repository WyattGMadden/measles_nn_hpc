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
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super(NeuralNetwork, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]

        # Append the specified number of hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Stack all layers in a Sequential module
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


def process_data(cases, year_test_cutoff):
    # Extract the year from the 'time' column
    cases['year'] = cases['time'].apply(lambda x: int(x))
    
    # Split data into training and test sets based on the cutoff year
    cases_train = cases[cases['year'] < year_test_cutoff]
    cases_test = cases[cases['year'] >= year_test_cutoff]

    # Prepare identifiers for training and test sets
    id_train = cases_train[['time', 'city']]
    id_test = cases_test[['time', 'city']]

    # Prepare target variables for training and test sets
    y_train = cases_train['cases'].to_numpy().reshape((-1, 1))
    y_test = cases_test['cases'].to_numpy().reshape((-1, 1))

    # Prepare feature matrices for training and test sets, dropping specific columns and filtering by regex
    drop_columns = ['time', 'city', 'cases', 'susc', 'pop', 'births', 'year']
    regex_filters = ['nbc', 'nearest_big_city', 'susc']

    X_train = cases_train.drop(columns=drop_columns)
    X_test = cases_test.drop(columns=drop_columns)

    for regex in regex_filters:
        X_train = X_train[X_train.columns.drop(list(X_train.filter(regex=regex)))]
        X_test = X_test[X_test.columns.drop(list(X_test.filter(regex=regex)))]

    # Create dataset objects for DataLoader
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    return train_data, test_data, X_train.shape[1], id_train, id_test



def get_dataloaders(train_data, test_data, batch_size):
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, pin_memory = True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory = True)

    return train_dataloader, test_dataloader



def train(model, device, train_loader, optimizer, loss_fn, epoch, log_interval, dry_run = False):
    model.train()
    train_loss = 0
    for dataloader_iter, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if dataloader_iter % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, dataloader_iter * len(X), len(train_loader.dataset),
                100. * dataloader_iter / len(train_loader), loss.item()))
            if dry_run:
                break       
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    return train_loss

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()  


    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss, len(test_loader.dataset) * test_loader.batch_size))

    test_loss /= len(test_loader.dataset)
    return test_loss
