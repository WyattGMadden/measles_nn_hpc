import argparse
import sys

sys.path.append('../data_processing/')
import prevac_measles_data_loader as mdl

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import full_basic_functions as fbf



def main():
    parser = argparse.ArgumentParser(description='Basic NN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BN',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--year-test-cutoff', type=float, default=61,
                        help='proportion of data for test')
    parser.add_argument('--num-hidden-layers', type=int, default=3,
                        help='Number of hidden layers in neural network')
    parser.add_argument('--hidden-dim', type=int, default=721,
                        help='Dimension of each hidden layer in neural network')
    parser.add_argument('--save-data-loc', type=str, default=".",
                        help='location to save output')
    parser.add_argument('--cases-data-loc', 
                        type=str, 
                        default="../../../data/data_from_measles_competing_risks/inferred_cases_urban.csv",
                        help='location of data')
    parser.add_argument('--pop-data-loc', type=str, default="../../../data/data_from_measles_competing_risks/inferred_pop_urban.csv",
                        help='location of data')
    parser.add_argument('--coords-data-loc', type=str, default="../../../data/data_from_measles_competing_risks/coordinates_urban.csv",
                        help='location of data')
    parser.add_argument('--susc-data-loc', type=str, default="../../../output/data/tsir_susceptibles/tsir_susceptibles.csv",
                        help='location of data')
    parser.add_argument('--birth-data-loc', type=str, default="../../../data/data_from_measles_competing_risks/ewBu4464.csv",
                        help='location of data')
    parser.add_argument('--num-epochs', type=int, default=200, metavar='EN',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00022, metavar='LR',
                        help='learning rate (default: 0.00022)')
    parser.add_argument('--weight-decay', type=float, default=0.0314, metavar='WD',
                        help='weight decay (default: 0.0314)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='LN',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--top-12-cities', action='store_true', default=False,
                        help='print info')
    parser.add_argument('--output-lossplot', action='store_true', default=False,
                        help='print info')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print info')
    parser.add_argument('--t-lag', type=int, default=130, metavar='TL',
                        help='Number of lags in features')


    args = parser.parse_args()

    if args.verbose:
        print("#################\n Model " + str(args.k) + "\n#################")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)



    cases, transform_data = mdl.create_measles_data(k = args.k,
                                                    t_lag = args.t_lag,
                                                    cases_data_loc = args.cases_data_loc,
                                                    pop_data_loc = args.pop_data_loc,
                                                    coords_data_loc = args.coords_data_loc,
                                                    susc_data_loc = args.susc_data_loc,
                                                    birth_data_loc = args.birth_data_loc,
                                                    top_12_cities = args.top_12_cities,
                                                    verbose = args.verbose)
    


    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, args.year_test_cutoff)
    train_dataloader, test_dataloader = fbf.get_dataloaders(train_data = train_data,
                                                            test_data = test_data,
                                                            batch_size = 64)


    input_dim = num_features
    hidden_dim = args.hidden_dim
    output_dim = 1
    model = fbf.NeuralNetwork(
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_hidden_layers = args.num_hidden_layers,
            ).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay) # mse 0.881 (raytune)

    train_loss = []
    test_loss = []

    for epoch in range(1, args.num_epochs + 1):
        train_loss_i = fbf.train(
                model, 
                device, 
                train_dataloader, 
                optimizer, 
                loss_fn, 
                epoch, 
                log_interval = args.log_interval,
                dry_run = args.dry_run
                )
        train_loss.append(train_loss_i)
        test_loss_i = fbf.test(model, device, test_dataloader, loss_fn)
        test_loss.append(test_loss_i)

    if args.save_model:
        torch.save(model.state_dict(), args.save_data_loc + str(args.k) + "_model.pt")

    if args.output_lossplot:
        # Plot loss
        plt.plot(train_loss, label = "train")
        plt.plot(test_loss, label = "test")
        plt.legend()
        #save
        plt.savefig(args.save_data_loc + str(args.k) + "_loss.png")

        

    pred_train = model(train_data.X.to(device)).to("cpu").detach().numpy()
    if args.verbose:
        print("MSE train \n")
        print(np.mean((pred_train - train_data.y.detach().numpy())**2))
        print("\n")
    pred_test = model(test_data.X.to(device)).to("cpu").detach().numpy()
    if args.verbose:
        print("MSE test \n")
        print(np.mean((pred_test - test_data.y.detach().numpy())**2))
        print("\n")


    if args.save_model:

        id_train['train_test'] = 'train'
        id_test['train_test'] = 'test'
        output = pd.concat([id_train, id_test], ignore_index = True)
        output.insert(3, 'pred', np.concatenate([pred_train, pred_test]))
        output.insert(4, 'cases', np.concatenate([train_data.y.detach().numpy(), test_data.y.detach().numpy()]))
        output.to_parquet(args.save_data_loc + str(args.k) + "_output.parquet")
        transform_data.to_parquet(args.save_data_loc + str(args.k) + "_transform.parquet")


if __name__ == '__main__':
    main()






