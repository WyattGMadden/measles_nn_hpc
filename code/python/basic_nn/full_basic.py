import argparse
import sys

original_sys_path = sys.path.copy()
sys.path.append('../data_processing/')
import prevac_measles_data_loader as mdl
sys.path = original_sys_path

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import full_basic_functions as fbf



def main():
    parser = argparse.ArgumentParser(description='Basic NN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BN',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='proportion of data for test')
    parser.add_argument('--save-data-loc', type=str, default=".",
                        help='location to save output')
    parser.add_argument('--cases-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--pop-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--coords-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--susc-data-loc', type=str, default="../../../data/tsir_susceptibles/tsir_susceptibles.csv",
                        help='location of data')
    parser.add_argument('--birth-data-loc', type=str, default="../../../data/births/ewBu4464.csv",
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
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print info')

    args = parser.parse_args()

    if args.verbose:
        print("#################\n Model " + str(args.k) + "\n#################")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)



    cases, transform_data = mdl.create_measles_data(k = args.k,
                                                    t_lag = 130,
                                                    cases_data_loc = args.cases_data_loc,
                                                    pop_data_loc = args.pop_data_loc,
                                                    coords_data_loc = args.coords_data_loc,
                                                    susc_data_loc = args.susc_data_loc,
                                                    birth_data_loc = args.birth_data_loc,
                                                    top_12_cities = args.top_12_cities,
                                                    verbose = args.verbose)
    


    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, args.test_size)
    train_dataloader, test_dataloader = fbf.get_dataloaders(train_data = train_data,
                                                            test_data = test_data,
                                                            batch_size = 64)


    input_dim = num_features
    hidden_dim = int(np.floor((num_features + 1) * 2 / 3))
    output_dim = 1
    model = fbf.NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay) # mse 0.881 (raytune)


    for epoch in range(1, args.num_epochs + 1):
        fbf.train(
                model, 
                device, 
                train_dataloader, 
                optimizer, 
                loss_fn, 
                epoch, 
                log_interval = args.log_interval,
                dry_run = args.dry_run
                )
        fbf.test(model, device, test_dataloader, loss_fn)

    if args.save_model:
        torch.save(model.state_dict(), args.save_data_loc + str(args.k) + "_model.pt")

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






