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
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import full_basic_functions as fbf


def main():
    parser = argparse.ArgumentParser(description='Random Forest Regression')
    parser.add_argument('--n-estimators', type=int, default=100, help='number of trees in the forest (default: 100)')
    parser.add_argument('--k', type=int, default=52, help='step ahead prediction')
    parser.add_argument('--test-size', type=float, default=0.3, help='proportion of data for test')
    parser.add_argument('--save-data-loc', type=str, default=".", help='location to save output')
    parser.add_argument('--cases-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--pop-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--coords-data-loc', type=str, default=".",
                        help='location of data')
    parser.add_argument('--susc-data-loc', type=str, default="../../../data/tsir_susceptibles/tsir_susceptibles.csv", help='location of data')
    parser.add_argument('--birth-data-loc', type=str, default="../../../data/births/ewBu4464.csv", help='location of data')
    parser.add_argument('--random-state', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true', default=False, help='print info')
    parser.add_argument('--top-12-cities', action='store_true', default=False, help='print info')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    
    args = parser.parse_args()

    if args.verbose:
        print("#################\n Model " + str(args.k) + "\n#################")

    cases, transform_data = mdl.create_measles_data(k=args.k,
                                                    t_lag=130,
                                                    cases_data_loc = args.cases_data_loc,
                                                    pop_data_loc = args.pop_data_loc,
                                                    coords_data_loc = args.coords_data_loc,
                                                    susc_data_loc=args.susc_data_loc,
                                                    birth_data_loc=args.birth_data_loc,
                                                    top_12_cities=args.top_12_cities,
                                                    verbose=args.verbose)
    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, args.test_size)

    model = RandomForestRegressor(
                n_estimators=args.n_estimators, 
                random_state=args.random_state, 
                verbose=2
            )


    train_data_X = train_data.X.numpy()
    train_data_y = train_data.y.numpy().flatten()

    test_data_X = test_data.X.numpy()
    test_data_y = test_data.y.numpy().flatten()

    model.fit(train_data_X, train_data_y)

    pred_train = model.predict(train_data_X)
    pred_test = model.predict(test_data_X)

    if args.verbose:
        print("MSE train \n")
        print(mean_squared_error(train_data_y, pred_train))
        print("MSE test \n")
        print(mean_squared_error(test_data_y, pred_test))
        print("\n")

    if args.save_model:
        # Saving the model and outputs
        joblib.dump(model, args.save_data_loc + str(args.k) + "_rf_model.joblib")
        
        id_train['train_test'] = 'train'
        id_test['train_test'] = 'test'
        output = pd.concat([id_train, id_test], ignore_index=True)
        output.insert(3, 'pred', np.concatenate([pred_train, pred_test]))
        output.insert(4, 'cases', np.concatenate([train_data_y, test_data_y]))
        output.to_parquet(args.save_data_loc + str(args.k) + "_output.parquet")
        transform_data.to_parquet(args.save_data_loc + str(args.k) + "_transform.parquet")


if __name__ == '__main__':
    main()

