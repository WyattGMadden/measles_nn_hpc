import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from captum.attr import NeuronConductance
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation, ShapleyValueSampling



import numpy as np
import pandas as pd


import full_basic as fb


def main():
    parser = argparse.ArgumentParser(description='Basic NN Explain')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BN',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='proportion of data for test')
    parser.add_argument('--save-data-loc', type=str, default=".",
                        help='location to save output')
    parser.add_argument('--susc-data-loc', type=str, default="../../../data/tsir_susceptibles/tsir_susceptibles.csv",
                        help='location of data')
    parser.add_argument('--birth-data-loc', type=str, default="../../../data/births/ewBu4464.csv",
                        help='location of data')
    parser.add_argument('--num-epochs', type=int, default=1, metavar='EN',
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='LN',
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



    
    #read parquets
    X_train = pd.read_parquet("../../../../output/data/basic_nn/explain/" + str(args.k) + "_X_train.parquet")
    X_test = pd.read_parquet("../../../../output/data/basic_nn/explain/" + str(args.k) + "_X_test.parquet")

    id_train = pd.read_parquet("../../../../output/data/basic_nn/explain/" + str(args.k) + "_id_train.parquet")
    id_test = pd.read_parquet("../../../../output/data/basic_nn/explain/" + str(args.k) + "_id_test.parquet")

    num_features = X_train.shape[1]
    input_dim = num_features
    hidden_dim = int(np.floor((num_features + 1) * 2 / 3))
    output_dim = 1
    model = fb.NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load("../../../../output/models/basic_nn/" + str(args.k) + "_model.pt"))

    


    ############################################
    #################CAPTUM#####################
    ############################################

    #ig = IntegratedGradients(model)
    #ig_nt = NoiseTunnel(ig)
    #dl = DeepLift(model)
    #gs = GradientShap(model)
    #fa = FeatureAblation(model)
    svs = ShapleyValueSampling(model)


    test_input_tensor = torch.from_numpy(X_test.to_numpy()).type(torch.FloatTensor).to(device)
    train_input_tensor = torch.from_numpy(X_train.to_numpy()).type(torch.FloatTensor).to(device)

    #groups
    group_size = 130 - args.k + 1
    for i in X_test.columns:
        print(i)

    fa_groups = np.concatenate(
            (np.repeat([0], [group_size]), #case_lags
             np.array([1]), #pop
             np.array([2]), #birth
             np.repeat([3], [group_size]), #bc case lags and distances
             np.repeat([4], [group_size]), #bc case lags and distances
             np.repeat([5], [group_size]), #bc case lags and distances
             np.repeat([6], [group_size]), #bc case lags and distances
             np.repeat([7], [group_size]), #bc case lags and distances
             np.repeat([8], [group_size]), #bc case lags and distances
             np.repeat([9], [group_size]), #bc case lags and distances
             np.arange(3, 10),
             np.repeat([10], [(group_size + 1) * 10]))) #nbc case lags and distances
#             np.tile(np.concatenate((np.repeat([4], group_size), [5])), 10))) #nbc case lags and distances


    fa_groups = torch.tensor(fa_groups).to(device)
    distinct_columns = X_test.columns[np.unique(fa_groups.cpu(), return_index = True)[1]]

    svs_attr_test = svs.attribute(test_input_tensor, feature_mask = fa_groups)
    #flatten
    svs_attr_test = svs_attr_test.reshape(svs_attr_test.shape[0], -1)
    svs_attr_test.shape
    svs_pd = pd.DataFrame(svs_attr_test.to('cpu').detach().numpy())
    svs_pd.columns = X_test.columns
    svs_pd = svs_pd[distinct_columns]
    svs_pd.columns 
    svs_pd['city'] = id_test['city'].values
    svs_pd['time'] = id_test['time'].values
    svs_pd.to_csv(args.save_data_loc + str(args.k) + "_svs_explain_sep_high_pop_groups.csv")



    #ig_attr_test = ig.attribute(test_input_tensor, train_input_tensor, feature_mask = fa_groups)
    #ig_attr_test.shape
    #ig_pd = pd.DataFrame(gs_attr_test.numpy())
    #ig_pd.columns = X_test.columns

    #gs_attr_test = gs.attribute(test_input_tensor, train_input_tensor)
    #gs_attr_test.shape
    #gs_pd = pd.DataFrame(gs_attr_test.numpy())
    #gs_pd.columns = X_test.columns
    #gs_pd.to_csv("../../../../output/data/basic_nn/explain/" + str(args.k) + "_gs_explain_dist_collated.csv")

    #fa_attr_test = fa.attribute(test_input_tensor, feature_mask = fa_groups)
    #fa_attr_test.shape
    #fa_pd = pd.DataFrame(fa_attr_test.numpy())
    #fa_pd.columns = X_test.columns




    #ig_nt_attr_test = ig_nt.attribute(test_input_tensor)
    #dl_attr_test = dl.attribute(test_input_tensor)
    #gs_attr_test = gs.attribute(test_input_tensor, train_input_tensor)
    #no groups
    #fa_attr_test = fa.attribute(test_input_tensor)

    #get distinct columns
    #distinct_columns = X_test.columns[np.unique(fa_groups, return_index = True)[1]]


    #fa_pd = fa_pd[distinct_columns]

    #fa_pd['city'] = id_test['city'].values
    #fa_pd['time'] = id_test['time'].values


    #fa_pd.to_csv("../../../../output/data/basic_nn/explain/" + str(args.k) + "_fa_explain_dist_collated.csv")

if __name__ == '__main__':
    main()

