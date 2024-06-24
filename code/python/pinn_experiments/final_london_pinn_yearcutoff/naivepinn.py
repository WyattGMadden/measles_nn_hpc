import argparse
import os

import math
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader






def readin_data(k = 52):
    train_df = pd.read_parquet("../../../../output/data/train_test_k/k" + str(k) + "_train.gzip")
    test_df = pd.read_parquet("../../../../output/data/train_test_k/k" + str(k) + "_test.gzip")
    return train_df, test_df

def get_cities(data, cities):
    data_cities = data[data['city'].isin(cities)].copy()
    return data_cities

def process(data, time_unit = 365):
    data['time'] = np.round((data['time'] - 51) * 26 + 1)
    data['births'] = data['births'] / 26
    return data

def get_data(data):
    #select multiple columns from data
    case_cols = [col for col in data.columns if "cases_lag_" in col]
    return data[["time", "susc", "cases", "births", "pop"] + case_cols]
                 




def get_X_y(data):
    S = data['susc'].to_numpy()
    S = S.reshape((S.shape[0], 1))
    I = data['cases'].to_numpy()
    I = I.reshape((I.shape[0], 1))
    t = data['time'].to_numpy()
    t = t.reshape((t.shape[0], 1))
    N = data['pop'].to_numpy()
    N = N.reshape((N.shape[0], 1))
    Bi = data['births'].to_numpy()
    Bi = Bi.reshape((Bi.shape[0], 1))
    X = data.drop(columns = ['cases', 'time', 'susc', 'pop', 'births']).to_numpy()
    return S, I, t, N, Bi, X







class Data(Dataset):
    def __init__(self, t, S, I, Bi, N, X):
        self.t = torch.from_numpy(t).float().reshape(1, -1).t()
        self.t_ode = torch.from_numpy(t).float().reshape(1, -1).t()
        self.S = torch.from_numpy(S).float().reshape(1, -1).t()
        self.I = torch.from_numpy(I).float().reshape(1, -1).t()
        self.Bi = torch.from_numpy(Bi).float().reshape(1, -1).t()
        self.N = torch.from_numpy(N).float().reshape(1, -1).t()
        self.X = torch.from_numpy(X).float()
        self.len = self.t.shape[0]
       
    def __getitem__(self, index):
        return self.t[index], self.t_ode[index], self.S[index], self.I[index], self.Bi[index], self.N[index], self.X[index]

   
    def __len__(self):
        return self.len




class fourier_map(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, B):
        super().__init__()
        self.size_B = B.shape[1]
        self.B = B
        self.size_in = size_in * self.size_B * 2
        self.size_out = self.size_in
        weights = torch.Tensor(self.size_out, self.size_in)
        self.weights = nn.Parameter(weights)  
        bias = torch.Tensor(self.size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        x_sin = torch.sin(torch.mm(x, self.B))
        x_cos = torch.cos(torch.mm(x, self.B))
        x = torch.cat((x_sin, x_cos), 1)
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b 


def mod_beta_torch(t, beta, T):
    which_beta = (t.detach().long() - 1) % 26
    return beta[which_beta]

#def seasonal_beta_torch(t, vert, amp, phase, T):
#    return torch.exp(vert) + torch.exp(amp) * torch.cos(2 * torch.pi * ((t / T) + phase))

#def seasonal_beta_torch(t, vert, amp, phase, T):
#    return vert + amp * torch.cos(2 * torch.pi * ((t / T) + phase))

def seasonal_beta_torch(t, vert, amp1, amp2, T):
    return vert + amp1 * torch.sin(2 * torch.pi * ((t / T))) + amp2 * torch.cos(2 * torch.pi * ((t / T)))


class derivative_layer(nn.Module):
    def __init__(self, T, num_t):
        super().__init__()
        self.T = T
        self.vert = nn.Parameter(torch.randn(1))
        self.amp1 = nn.Parameter(torch.randn(1))
        self.amp2 = nn.Parameter(torch.randn(1))
        self.S_latent = nn.Parameter(torch.randn(num_t))



    def forward(self, t, SI, Bi, N):
        #S = torch.index_select(SI, 1, torch.tensor([0]))
        S_lat = torch.exp(self.S_latent[(t.long() - 1)] + 4.5) * 1e3
        I = SI[:, 1:2]
        beta = seasonal_beta_torch(t, self.vert, self.amp1, self.amp2, self.T)
        der_S = Bi - ((beta) * S_lat * I) / N
        der_I = ((beta) * S_lat * I) / N - (1) * I
        der_SI = torch.cat((der_S, der_I), 1)
        return der_SI

class ode_nn(nn.Module):
    def __init__(self, input_dim,  output_dim, T, num_t):
        super(ode_nn, self).__init__()
        self.der = derivative_layer(T, num_t)
    def forward(self, t, SI, Bi, N):
        x = self.der(t, SI, Bi, N)
        return x          

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim_t, input_dim_X, hidden_dim, output_dim, B):
        super(NeuralNetwork, self).__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.gelu = nn.GELU()
        self.fm = fourier_map(input_dim_t, B)
        self.f1 = nn.Linear(B.shape[1] * 2 + input_dim_X, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        self.f6 = nn.Linear(hidden_dim, output_dim)
    def forward(self, t, X):
        t_B = self.gelu(self.fm(t))
        x = self.gelu(self.f1(torch.cat((t_B, X), dim = 1)))
        x = self.gelu(self.f2(x))
        x = self.gelu(self.f3(x))
        x = self.f6(x)
        x = self.softplus(x)
        return x

def get_B(num_features = 50, scale = 1):
    B = np.random.randn(1, num_features) * scale  # Random basis
    B = torch.FloatTensor(B)
    return B








def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=52, help="step ahead prediction")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    np.random.seed(42)

    train_df, test_df = readin_data(k = args.k)
    top_cities = ["London"]
    train_london = get_cities(train_df, top_cities)
    test_london = get_cities(test_df, top_cities)
    cases_train = get_data(process(train_london))
    cases_test = get_data(process(test_london))
    S_train, I_train, t_train, N_train, Bi_train, X_train = get_X_y(cases_train)
    S_test, I_test, t_test, N_test, Bi_test, X_test = get_X_y(cases_test)
    train_data = Data(t = t_train,
                      S = S_train,
                      I = I_train,
                      Bi = Bi_train,
                      N = N_train,
                      X = X_train)

    test_data = Data(t = t_test,
                     S = S_test,
                     I = I_test,
                     Bi = Bi_test,
                     N = N_test,
                     X = X_test)



    batch_size = 64
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True)



    input_dim_t = 1
    input_dim_X = train_data.X.shape[1]
    hidden_dim = 128
    output_dim = 3
    B = get_B(scale = 0.1).to(device)

    model = NeuralNetwork(input_dim_t = input_dim_t,
                          input_dim_X = input_dim_X,
                          hidden_dim = hidden_dim, 
                          output_dim = output_dim,
                          B = B).to(device)


    ode_model = ode_nn(input_dim = 3, 
                       output_dim = 3,
                       T = 26,
                       num_t = t_train.shape[0]).to(device)

    loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.01)
    optimizer_ode = torch.optim.Adam(ode_model.parameters(), lr=0.1)
    num_epochs = 10000
    ode_loss_values = []
    S_loss_values = []
    I_loss_values = []

    S_test_loss_values = []
    I_test_loss_values = []

    vert_params = []
    amp1_params = []
    amp2_params = []
    S_lat_params = []
    S_grad_values_all = []
    I_grad_values_all = []

    train_loss = 0
    model.train()
    ode_model.train()

    S_hp = 1 / 10
    I_hp = 1 * 10
    ode_hp = 1 * 1

    i = 0
    for epoch in range(num_epochs):
        train_S_loss = 0
        train_I_loss = 0
        train_ode_loss = 0
        train_S_grad = 0
        train_I_grad = 0
        S_grad_values = []
        I_grad_values = []


        for t, t_ode, S, I, Bi, N, X, in train_dataloader:
            #all to device
            t = t.to(device)
            t_ode = t_ode.to(device)
            S = S.to(device)
            I = I.to(device)
            Bi = Bi.to(device)
            N = N.to(device)
            X = X.to(device)

            S.requires_grad = True
            I.requires_grad = True
            t.requires_grad = True

            # zero the parameter gradient
            optimizer.zero_grad()
            optimizer_ode.zero_grad()

            # forward + backward + optimize
            pred = model(t, X)
            S_pred = pred[:, 0:1]
            I_pred = pred[:, 1:2]

            #get grad
            u_x = torch.autograd.functional.jacobian(model, 
                                                     (t, X),
                                                     create_graph=True)

            u_t = u_x[0]
            u_s = u_t[:, 0:1]
            u_s = torch.squeeze(u_s, 1)
            u_s = torch.diagonal(u_s)
            u_i = u_t[:, 1:2]
            u_i = torch.squeeze(u_i, 1)
            u_i = torch.diagonal(u_i)

            grad_for_loss_S = torch.reshape(u_s, (-1, 1))
            S_grad_values.append(grad_for_loss_S.detach().to("cpu").numpy()[:, 0])

            grad_for_loss_I = torch.reshape(u_i, (-1, 1))
            I_grad_values.append(grad_for_loss_I.detach().to("cpu").numpy()[:, 0])


            der = ode_model(t = t, SI = pred, Bi = Bi, N = N)
            der_S = der[:, 0:1]
            der_I = der[:, 1:2]


            #get latent parameters for this batch of data
            S_latent_batch = list(ode_model.parameters())[3][t.long() - 1]




            loss_S = loss_fn(S_pred, S_latent_batch)
            loss_I = loss_fn(I_pred, I)

            loss_grad_S = loss_fn(der_S, grad_for_loss_S)
            loss_grad_I = loss_fn(der_I, grad_for_loss_I)

            loss_ode = loss_grad_S + loss_grad_I


            loss = loss_S * S_hp + loss_I * I_hp  + loss_ode * ode_hp
            loss.retain_grad()

            train_S_loss += loss_S.item()
            train_I_loss += loss_I.item()
            train_ode_loss += loss_ode.item()

            loss.backward()
            #loss = loss_cases + loss_grad_S + loss_grad_I
            optimizer.step()
            optimizer_ode.step()
            params_all = list(ode_model.parameters())

        print("loss:")
        print(loss.detach().to("cpu").numpy().flatten())
        print("\n")
        print("lossS:")
        print(loss_S.detach().to("cpu").numpy().flatten()* S_hp)
        print("\n")
        print("lossI:")
        print(loss_I.detach().to("cpu").numpy().flatten() * I_hp)
        print("\n")
        print("lossODE:")
        print(loss_ode.detach().to("cpu").numpy().flatten() * ode_hp)
        print("\n")
        print("vert:")
        vert_params.append(params_all[0].to("cpu").detach().numpy().flatten())
        print(vert_params[i])
        print("\n")
        print("amp1:")
        amp1_params.append(params_all[1].to("cpu").detach().numpy().flatten())
        print(amp1_params[i])
        print("\n")
        print("amp2:")
        amp2_params.append(params_all[2].to("cpu").detach().numpy().flatten())
        print(amp2_params[i])
        print("\n")
        print("\n")
        print("S_lat:")
        S_lat_params.append(params_all[3].to("cpu").detach().numpy().flatten())
        print(S_lat_params[i])
        print("\n")
        print("\n")
        S_grad_values_all.append(np.concatenate(S_grad_values))
        I_grad_values_all.append(np.concatenate(I_grad_values))
        with torch.no_grad():
            test_pred = model(test_data.t.to(device), test_data.X.to(device)).to("cpu")
            S_pred = test_pred[:, 0:1]
            I_pred = test_pred[:, 1:2]
            S_test_loss_values.append(loss_fn(S_pred, test_data.S).item())
            I_test_loss_values.append(loss_fn(I_pred, test_data.I).item())

            


        i += 1
        print(i)

        ode_loss_values.append(train_ode_loss)
        S_loss_values.append(train_S_loss)
        I_loss_values.append(train_I_loss)





    loc_name = "naivepinn"
    write_loc = "../../../../output/models/pinn_experiments/final_london_pinn_yearcutoff/"

    #os.mkdir('../../../../output/models/pinn_experiments/' + loc_name)


    torch.save(model.state_dict(), 
               write_loc + loc_name + "_feature_model.pt")
    torch.save(ode_model.state_dict(), 
               write_loc + loc_name + "_ode_model.pt")


    # create pd with all by-iter results
    df = pd.DataFrame()
    df['ode_loss'] = ode_loss_values
    df['S_loss'] = S_loss_values
    df['I_loss'] = I_loss_values
    df['S_test_loss'] = S_test_loss_values
    df['I_test_loss'] = I_test_loss_values
    df['vert'] = vert_params
    df['amp1'] = amp1_params
    df['amp2'] = amp2_params

    S_lat_params = pd.DataFrame(S_lat_params)
    #add "S" to front  of column names
    S_lat_params.columns = ['S_' + str(col) for col in S_lat_params.columns]

    #concatenate pd data frames by column
    df = pd.concat([df, S_lat_params], axis=1)

    #add "beta" to all column names
    #add dfbeta to df

    # save as parquet
    df.to_parquet(write_loc + loc_name + "_fit_info.parquet")

    # save train/test predictions
    model.eval()

    train_pred = model(train_data.t.to(device), train_data.X.to(device))
    train_pred = train_pred.detach().to("cpu").numpy()
    train_pred = pd.DataFrame(train_pred)
    train_pred.columns = ['S_pred', 'I_pred', 'R_pred']
    train_pred['time'] = train_data.t.detach().numpy()
    train_pred['S'] = S_train
    train_pred['I'] = I_train
    train_pred.to_parquet(write_loc + loc_name + "_train_predictions.parquet")


    test_pred = model(test_data.t.to(device), test_data.X.to(device))
    test_pred = test_pred.detach().to("cpu").numpy()
    test_pred = pd.DataFrame(test_pred)
    test_pred.columns = ['S_pred', 'I_pred', 'R_pred']
    test_pred['time'] = test_data.t.detach().numpy()
    test_pred['S'] = S_test
    test_pred['I'] = I_test
    test_pred.to_parquet(write_loc + loc_name + "_test_predictions.parquet")





if __name__ == "__main__":
    main()
