import argparse
import numpy as np
import pandas as pd
import torch
np.random.seed(2)

def get_train_test(dat, wrt, test_size):
    #split train/test
    dat_train = dat[dat[wrt] <= dat[wrt].quantile(1 - test_size)]
    dat_test = dat[dat[wrt] > dat[wrt].quantile(1 - test_size)]
    return dat_train, dat_test


def create_measles_data(k, t_lag, test_size, birth_data_loc, susc_data_loc, write_to_file = False, top_12_cities = False, verbose = False, include_nbc_cases = False):

    cases = pd.read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/inferred_cases_urban.csv").rename(columns={"Unnamed: 0": "time"})

    population = pd.read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/inferred_pop_urban.csv").rename(columns={"Unnamed: 0": "time"})

    coords = pd.read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/coordinates_urban.csv", index_col = 0).rename(columns={"Unnamed: 0": "time"}).T

    births = pd.read_csv(birth_data_loc, index_col = 1).rename(columns={"Unnamed: 0": "year"})

    susc_df = pd.read_csv(susc_data_loc)


    if top_12_cities:
        top_n_city_names = population.iloc[0, 1:].nlargest(12).index
        cases = cases[top_n_city_names.insert(0, 'time')]
        population = population[top_n_city_names.insert(0, 'time')]
        coords = coords.loc[top_n_city_names]
        susc_df = susc_df[susc_df['city'].isin(top_n_city_names)]
        births = births[[col for col in top_n_city_names.insert(0, 'year').tolist() if col in births.columns]]



    #cases
    cases_long = pd.melt(cases, 
                         id_vars = 'time',
                         var_name = 'city',
                         value_name = 'cases')
    cases_groups = cases_long.groupby(['city'])
    cases_mean, cases_std = cases_groups.transform("mean"), cases_groups.transform("std")
    cases_long['cases_trans'] = (cases_long['cases'] - cases_mean['cases']) / cases_std['cases']

    for i in range(k, t_lag + 1):
        cases_long["cases_lag_" + str(i)] = cases_long.groupby(["city"])['cases_trans'].shift(i)

    cases_long.drop(columns = 'cases_trans', inplace = True)

    #population
    pop_long = pd.melt(population, 
                       id_vars = 'time',
                       var_name = 'city',
                       value_name = 'pop')
    pop_long['pop_std'] = (pop_long['pop'] - pop_long['pop'].mean()) / pop_long['pop'].std()
    #most recent birth of same season that doesn't cause data leakage
    k_temp = ((k + 25) // 26) * 26
    pop_long["pop_lag_" + str(k_temp)] = pop_long["pop_std"].shift(k_temp)
    pop_long.drop(columns = 'pop_std', inplace = True)

    cases_long = cases_long.merge(pop_long, 
            how = 'left', 
            on = ['time', 'city'])


    #reconstructed susceptibles
    susc_df['time'] = np.round(susc_df['time'] - 1900, 5)
    susc_df.drop(columns = ['births', 'pop', 'cases'], inplace = True)

    susc_groups = susc_df.groupby(['city'])
    susc_mean, susc_std = susc_groups.transform("mean"), susc_groups.transform("std")
    susc_df['susc_trans'] = (susc_df['susc'] - susc_mean['susc']) / susc_std['susc']
    #within-group mean imputation for missing values
    susc_df['susc_trans'] = np.where(pd.isna(susc_df['susc_trans']), 
                                     susc_mean['susc'], 
                                     susc_df['susc_trans'])


    for i in range(k, t_lag + 1):
        susc_df["susc_lag_" + str(i)] = susc_df.groupby(["city"])["susc_trans"].shift(i)

    susc_df.drop(columns = 'susc_trans', inplace = True)
    cases_long = cases_long.merge(susc_df,
                                  how = 'left', 
                                  on = ['time', 'city'])


    #births
    #within-city mean imputation
    births = births.fillna(births.mean())
    births = np.round(births)
    births_long = pd.melt(births, 
            id_vars = 'year',
            var_name = 'city',
            value_name = 'births')
    births_long['births_std'] = (births_long['births'] - births_long['births'].mean()) / births_long['births'].std()

    cases_long['year'] = np.floor(cases_long['time']).astype(int)
    cases_long = cases_long.merge(births_long, how = 'left', on = ['year', 'city'])
    cases_long.drop(columns = 'year', inplace = True)

    #replace missing values due to all zeros (std = 0) with 0
    cases_long['births_std'] = np.where(pd.isna(cases_long['births_std']),
                                         0,
                                         cases_long['births_std'])

    #most recent birth of same season that doesn't cause data leakage
    k_temp = ((k + 25) // 26) * 26
    cases_long["births_lag_" + str(k_temp)] = cases_long.groupby(["city"])["births_std"].shift(k_temp)
    
    cases_long.drop(columns = 'births_std', inplace = True)

    #big city cases as a feature
    top_7_cities = population.iloc[0, 1:].nlargest(7).index

    for i in top_7_cities:

        temp_cases_all = cases_long[cases_long['city'] == i]
        temp_cases = temp_cases_all.filter(regex = "cases_lag_.*", axis = 1)
        temp_cases = temp_cases.rename(columns = lambda x: x.replace("cases_lag_", "cases_" + i.lower() + "_lag_"))
        temp_cases['time'] = temp_cases_all['time']
        cases_long = cases_long.merge(temp_cases, how = 'left', on = ['time'])
        if verbose:
            print(i + " cases joined")

    #distance to each big city
    def spherical_dist(pos1, pos2, r=3958.75):
        pos1 = pos1 * np.pi / 180
        pos2 = pos2 * np.pi / 180
        cos_lat1 = np.cos(pos1[..., 0])
        cos_lat2 = np.cos(pos2[..., 0])
        cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
        cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
        return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


    coord_dist = spherical_dist(coords.values[:, None], coords.values)

    for i in top_7_cities:
        city_i_ind = np.where(i == coords.index)[0][0]
        city_i_dist = coord_dist[city_i_ind]
        pd_temp = pd.DataFrame(data = {'city': coords.index,
                                       'dist_' + i.lower(): city_i_dist})
        pd_temp['dist_' + i.lower()] = (pd_temp['dist_' + i.lower()] - pd_temp['dist_' + i.lower()].mean()) / pd_temp['dist_' + i.lower()].std()
        cases_long = cases_long.merge(pd_temp, how = 'left', on = ['city'])
        if verbose:
            print("distance to " + i + " joined")



    

    ###########################
    ###nearest big city lags###
    ###########################
    


    #get large city indices
    city_locs = []
    for i in range(top_7_cities.shape[0]):
        city_locs.append(np.where(coords.index == top_7_cities[i])[0][0])

    #get nearest large city index and distances for every unique city
    nearest_big_city_locs = []
    nearest_big_city_distances = []
    for i in range(coord_dist.shape[0]):
        distances = coord_dist[i][city_locs]
        #get index of nbc that is minimum distance from city (but not distance of zero, ie the same city)
        min_distance_index = np.where(distances == np.min(distances[distances > 0]))[0][0]
        nearest_big_city_distances.append(distances[min_distance_index])
        nearest_big_city_locs.append(city_locs[min_distance_index])

    nearest_big_city = coords.index[nearest_big_city_locs]

    #get nearest big city and distances for every city 
    nearest_big_city_ordered = [None] * cases_long.shape[0]
    for i in range(cases_long.shape[0]):
        nearest_big_city_ordered[i] = (np.where(cases_long['city'][i] == coords.index)[0][0])

    nearest_big_city = nearest_big_city[nearest_big_city_ordered]
    nearest_big_city_distances = np.array(nearest_big_city_distances)[nearest_big_city_ordered]


    #join on nearest big city lagged cases
    cases_long['nearest_big_city'] = nearest_big_city

    #join on distances to nearest big city
    cases_long['nearest_big_city_distances_unscaled'] = nearest_big_city_distances 
    cases_long['nearest_big_city_distances'] = (nearest_big_city_distances - np.mean(nearest_big_city_distances)) / np.std(nearest_big_city_distances)

    lag_vec = ["cases_lag_" + str(i) for i in range(k, t_lag + 1)]
    if include_nbc_cases:
        to_join_columns = np.concatenate((['time', 'city', 'cases'], lag_vec))
    else:
        to_join_columns = np.concatenate((['time', 'city'], lag_vec))

    cases_long_to_join = cases_long[to_join_columns].copy()
    cases_long_to_join.columns = (x.replace('cases_lag_', 'cases_nbc_lag_') for x in cases_long_to_join.columns)

    #rename cases and nbc_cases
    if include_nbc_cases:
        cases_long_to_join.rename(columns = {'cases':'nbc_cases'}, inplace = True)
        cases_long_to_join.rename(columns = {'city':'nearest_big_city'}, inplace = True)
    else:
        cases_long_to_join.rename(columns = {'city':'nearest_big_city'}, inplace = True)
        cases_long = cases_long.drop('nearest_big_city', axis = 1)


    cases_long = cases_long.merge(cases_long_to_join, 
                                  how = 'left', 
                                  on = ['time', 'nearest_big_city'])



    ##############################################
    #####get nearest 10 cities (not big) lags#####
    ##############################################

    coord_dist_pd = pd.DataFrame(coord_dist, columns = coords.index)
    coord_dist_pd['city_a'] = coords.index
    coord_dist_long = coord_dist_pd.melt(id_vars = ['city_a'], var_name = 'city_b', value_name = 'dist')


    nearest_10_city_indices = np.argsort(coord_dist, axis = 1)[:, 1:11]
    coords_names = np.array(coords.index)
    def get_city_names(x, axis):
        return coords_names[x]

    nearest_10_city_indices = np.apply_over_axes(get_city_names, 
                                                 nearest_10_city_indices, 
                                                 [0])
    nearest_10_cities_pd = pd.DataFrame(nearest_10_city_indices, 
                                        columns = np.char.array(['nearest_']) + np.arange(1, 11).astype(str) + np.char.array(['_city']))
    nearest_10_cities_pd['city'] = coords.index

    cases_long = cases_long.merge(nearest_10_cities_pd, how = 'left', on = ['city'])

    lag_vec = ["cases_lag_" + str(i) for i in range(k, t_lag + 1)]
    to_join_columns = np.concatenate((['time', 'city'], lag_vec))


    cases_long_to_join = cases_long[to_join_columns]

    for j in range(1, 11):
        cases_long_to_join_temp = cases_long_to_join.copy()
        cases_long_to_join_temp.columns = (x.replace('cases_lag_', 'cases_nc_' + str(j) + '_lag_') for x in cases_long_to_join_temp.columns)
        cases_long_to_join_temp.rename(columns = {'city':'nearest_' + str(j) + '_city'}, inplace = True)


        cases_long = cases_long.merge(cases_long_to_join_temp, 
                                      how = 'left', 
                                      on = ['time', 'nearest_' + str(j) + '_city'])

        cases_long = cases_long.merge(coord_dist_long,
                                      how = 'left',
                                      left_on = ['city', 'nearest_' + str(j) + '_city'],
                                      right_on = ['city_a', 'city_b'])
        cases_long.drop(['city_a', 'city_b'], axis = 1, inplace = True)
        cases_long['dist'] = (cases_long['dist'] - np.mean(cases_long['dist'])) / np.std(cases_long['dist'])
        cases_long.rename(columns = {'dist':'nearest_' + str(j) + '_city_dist'}, inplace = True)
        cases_long.drop('nearest_' + str(j) + '_city', axis = 1, inplace = True)
        

        if verbose:
            print('nearest ' + str(j) + ' city lags joined')












    cases_long = cases_long[cases_long.time < 65]

    #drop rows with missing values due to lags
    drop_times = cases.time[0:(k + t_lag)]
    cases_long = cases_long[~cases_long['time'].isin(drop_times)]


    cases_long_train, cases_long_test = get_train_test(dat = cases_long, wrt = 'time', test_size = test_size)

    if write_to_file:
        #cases_long.to_parquet("../../../output/data/grav_pinn_data/k" + str(k) + ".gzip", compression = "gzip")
        cases_long_train.to_parquet("../../../output/data/train_test_k/" + "k" + str(k) + "_train.gzip", compression = "gzip")
        cases_long_test.to_parquet("../../../output/data/train_test_k/" + "k" + str(k) + "_test.gzip", compression = "gzip")

    else:
        return cases_long_train, cases_long_test

   
def main():
    parser = argparse.ArgumentParser(description='Data loader for measles data')
    parser.add_argument('--k', type=int, default=52,
                        help='step ahead prediction')
    parser.add_argument('--t-lag', type=int, default=130,
                        help='number of lags')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='proportion of data for test')
    parser.add_argument('--susc-data-loc', type=str, default="../../../data/tsir_susceptibles/tsir_susceptibles.csv",
                        help='location of data')
    parser.add_argument('--birth-data-loc', type=str, default="../../../data/births/ewBu4464.csv",
                        help='location of data')
    parser.add_argument('--write-to-file', action='store_true',
                        help='whether to write to file')
    parser.add_argument('--include-nbc-cases', action='store_true',
                        help='whether to include nearest big city cases')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to print verbose')
    args = parser.parse_args()
    create_measles_data(k = args.k,
                        t_lag = args.t_lag,
                        test_size = args.test_size,
                        susc_data_loc = args.susc_data_loc,
                        birth_data_loc = args.birth_data_loc,
                        write_to_file = args.write_to_file,
                        include_nbc_cases = args.include_nbc_cases,
                        verbose = args.verbose)




if __name__ == "__main__":
    main()



