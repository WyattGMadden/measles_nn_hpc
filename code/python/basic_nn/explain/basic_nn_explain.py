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

import sys
sys.path.insert(0, '../')
import full_basic_functions as fbf
sys.path.insert(0, 'explain/')



def main():
    parser = argparse.ArgumentParser(description='Basic NN Explain')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BN',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--t-lag', type=int, default=130,
                        help='feature lags.')
    parser.add_argument('--hidden-dim', type=int, default=721,
                        help='Hidden dimension.')
    parser.add_argument('--num-hidden-layers', type=int, default=2,
                        help='Number of hidden layers.')
    parser.add_argument(
            '--write-data-loc', 
            type=str, 
            default="../../../../output/data/basic_nn_yearcutoff_optimal/explain/",
            help='location of data'
            )
    parser.add_argument(
            '--model-read-loc', 
            type=str, 
            default="../../../../output/models/basic_nn_yearcutoff_optimal/",
            help='location of data'
            )
    parser.add_argument(
            '--data-read-loc', 
            type=str, 
            default="../../../../output/data/basic_nn_yearcutoff_optimal/explain/",
            help='location of data'
            )
    parser.add_argument(
            '--susc-data-loc', 
            type=str, 
            default="../../../../output/data/tsir_susceptibles/tsir_susceptibles.csv",
            help='location of data'
            )
    parser.add_argument(
            '--birth-data-loc', 
            type=str, 
            default="../../../../data/data_from_measles_competing_risks/ewBu4464.csv",
            help='location of data'
            )
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print info')

    args = parser.parse_args()

    if args.verbose:
        print("#################\n Model " + str(args.k) + "\n#################")
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(42)
    
    device = torch.device("cuda" if use_cuda else "cpu")



    
    #read parquets
    X_train = pd.read_parquet(args.data_read_loc + str(args.k) + "_X_train.parquet")
    X_test = pd.read_parquet(args.data_read_loc + str(args.k) + "_X_test.parquet")

    id_train = pd.read_parquet(args.data_read_loc + str(args.k) + "_id_train.parquet")
    id_test = pd.read_parquet(args.data_read_loc + str(args.k) + "_id_test.parquet")

    num_features = X_train.shape[1]
    input_dim = num_features
    hidden_dim = args.hidden_dim
    output_dim = 1
    model = fbf.NeuralNetwork(input_dim, hidden_dim, output_dim, num_hidden_layers = args.num_hidden_layers).to(device)
    model.load_state_dict(torch.load(args.model_read_loc + str(args.k) + "_model.pt"))

    


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
    svs_pd.to_csv(args.write_data_loc + str(args.k) + "_svs_explain_sep_high_pop_groups.csv")




if __name__ == '__main__':
    main()

