import argparse
import sys
sys.path.insert(0, '../../data_processing/')
import prevac_measles_data_loader as mdl
sys.path.insert(0, '../basic_nn/explain/')


import numpy as np
import pandas as pd



def process_data(cases, year_test_cutoff):

    cases['cases'] = np.log(cases['cases'] + 1)
    cases_groups = cases.groupby(['city'])
    cases_mean, cases_std = cases_groups.transform("mean"), cases_groups.transform("std")
    cases_transform_output = cases[['time', 'city']].copy()
    cases_transform_output['cases_mean'] = cases_mean['cases']
    cases_transform_output['cases_std'] = cases_std['cases']

    cases['cases'] = (cases['cases'] - cases_mean['cases']) / cases_std['cases']

    cases['year'] = cases['time'].apply(lambda x: int(x))
    cases_train = cases[cases['year'] < year_test_cutoff]
    cases_test = cases[cases['year'] >= year_test_cutoff]

    id_train = cases_train[['time', 'city']]
    id_test = cases_test[['time', 'city']]

    y_train = cases_train['cases'].to_numpy()
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = cases_test['cases'].to_numpy()
    y_test = y_test.reshape((y_test.shape[0], 1))

    #X_train = train_df[abl_cols]
    X_train = cases_train
    X_train = X_train.drop(['time', 'year', 'city', 'cases', 'cases_trans', 'births', 'susc', 'pop'], axis = 1)
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nbc')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nearest_big_city')))]
    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='susc')))]
#    X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='nc')))]

    #X_test = test_df[abl_cols]
    X_test = cases_test
    X_test = X_test.drop(['time', 'year', 'city', 'cases', 'cases_trans', 'births', 'susc', 'pop'], axis = 1)
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nbc')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nearest_big_city')))]
    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='susc')))]
#    X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='nc')))]


    return X_train, y_train, X_test, y_test, id_train, id_test





def main():
    parser = argparse.ArgumentParser(description='Basic NN')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--t-lag', type=int, default=130,
                        help='time lag features')
    parser.add_argument('--year-test-cutoff', type=int, default=61,
                        help='Year to split train and test data')
    parser.add_argument('--save-data-loc', type=str, default="./",
                        help='location to save output')
    parser.add_argument('--cases-data-loc', 
                        type=str, 
                        default="../../../../output/data/basic_nn/prefit/",
                        help='location of cases data')

    args = parser.parse_args()


    
    #read gzip parquett
    cases = pd.read_parquet(args.cases_data_loc + "k" + str(args.k) + "_tlag" + str(args.t_lag) + ".gzip")


    X_train, y_train, X_test, y_test, id_train, id_test = process_data(cases, args.year_test_cutoff)

    #write to parquet
    X_train.to_parquet(args.save_data_loc + str(args.k) + "_X_train.parquet")
    X_test.to_parquet(args.save_data_loc + str(args.k) + "_X_test.parquet")

    id_train.to_parquet(args.save_data_loc + str(args.k) + "_id_train.parquet")
    id_test.to_parquet(args.save_data_loc + str(args.k) + "_id_test.parquet")


if __name__ == '__main__':
    main()






