import argparse
import sys
sys.path.insert(0, '../')
import data_load as mdl
sys.path.insert(0, 'explain/')


import numpy as np
import pandas as pd



def process_data(cases, test_size):

    cases['cases'] = np.log(cases['cases'] + 1)
    cases_groups = cases.groupby(['city'])
    cases_mean, cases_std = cases_groups.transform("mean"), cases_groups.transform("std")
    cases_transform_output = cases[['time', 'city']].copy()
    cases_transform_output['cases_mean'] = cases_mean['cases']
    cases_transform_output['cases_std'] = cases_std['cases']

    cases['cases'] = (cases['cases'] - cases_mean['cases']) / cases_std['cases']

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
    X_train = X_train.drop(['time', 'city', 'cases', 'births', 'susc', 'pop'], axis = 1)
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nbc')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nearest_big_city')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='susc')))]
#    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nc')))]

    #X_test = test_df[abl_cols]
    X_test = cases_test
    X_test = X_test.drop(['time', 'city', 'cases', 'births', 'susc', 'pop'], axis = 1)
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nbc')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nearest_big_city')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='susc')))]
#    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nc')))]


    return X_train, y_train, X_test, y_test, id_train, id_test





def main():
    parser = argparse.ArgumentParser(description='Basic NN')
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
    parser.add_argument('--top-12-cities', action='store_true', default=False,
                        help='print info')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print info')

    args = parser.parse_args()


    
    cases, cases_transform_output = mdl.create_measles_data(k = args.k,
                                    t_lag = 130,
                                    susc_data_loc = args.susc_data_loc,
                                    birth_data_loc = args.birth_data_loc,
                                    top_12_cities = args.top_12_cities,
                                    verbose = args.verbose)

    X_train, y_train, X_test, y_test, id_train, id_test  = process_data(cases, args.test_size)

    #write to parquet
    X_train.to_parquet(args.save_data_loc + str(args.k) + "_X_train.parquet")
    X_test.to_parquet(args.save_data_loc + str(args.k) + "_X_test.parquet")

    id_train.to_parquet(args.save_data_loc + str(args.k) + "_id_train.parquet")
    id_test.to_parquet(args.save_data_loc + str(args.k) + "_id_test.parquet")


if __name__ == '__main__':
    main()






