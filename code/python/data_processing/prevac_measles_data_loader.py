
import numpy as np
import pandas as pd
import urllib.request
import torch
np.random.seed(2)




def create_measles_data(
        k, 
        t_lag, 
        cases_data_loc,
        pop_data_loc,
        coords_data_loc,
        birth_data_loc, 
        susc_data_loc, 
        write_to_file = False, 
        top_12_cities = False, 
        verbose = False, 
        current_births = False
        ):

    cases = pd.read_csv(cases_data_loc).rename(columns={"Unnamed: 0": "time"})

    population = pd.read_csv(pop_data_loc).rename(columns={"Unnamed: 0": "time"})

    coords = pd.read_csv(coords_data_loc, index_col = 0).rename(columns={"Unnamed: 0": "time"}).T

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
    cases_long['cases'] = np.log(cases_long['cases'] + 1)

    cases_groups = cases_long.groupby(['city'])
    cases_mean, cases_std = cases_groups.transform("mean"), cases_groups.transform("std")
    cases_long['cases'] = (cases_long['cases'] - cases_mean['cases']) / cases_std['cases']

    cases_transform_output = cases_long[['time', 'city']].copy()
    cases_transform_output['cases_mean'] = cases_mean['cases']
    cases_transform_output['cases_std'] = cases_std['cases']

    # Create a list to hold the temporary DataFrames
    temp_dfs = []

    # Iterate and create lagged DataFrames
    for i in range(k, t_lag + 1):
        lag_df = cases_long.groupby('city')['cases'].shift(i)
        lag_df.name = "cases_lag_" + str(i)
        temp_dfs.append(lag_df)

    # Concatenate all the lagged DataFrames column-wise
    lagged_data = pd.concat(temp_dfs, axis=1)

    cases_long = cases_long.join(lagged_data)

    
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


    # Create a list to hold the temporary DataFrames
    temp_dfs = []

    # Iterate and create lagged DataFrames
    for i in range(k, t_lag + 1):
        lag_df = susc_df.groupby('city')['susc_trans'].shift(i)
        lag_df.name = "susc_lag_" + str(i)
        temp_dfs.append(lag_df)

    # Concatenate all the lagged DataFrames column-wise
    lagged_data = pd.concat(temp_dfs, axis=1)

    susc_df = susc_df.join(lagged_data)

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

    if current_births:
        cases_long["births_lag_0"] = cases_long["births_std"]
    
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
        min_distance_index = np.where(distances == np.min(distances))[0][0]
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
    cases_long['nearest_big_city_distances'] = (nearest_big_city_distances - np.mean(nearest_big_city_distances)) / np.std(nearest_big_city_distances)

    lag_vec = ["cases_lag_" + str(i) for i in range(k, t_lag + 1)]
    to_join_columns = np.concatenate((['time', 'city'], lag_vec))

    cases_long_to_join = cases_long[to_join_columns].copy()
    cases_long_to_join.columns = (x.replace('cases_lag_', 'cases_nbc_lag_') for x in cases_long_to_join.columns)
    cases_long_to_join.rename(columns = {'city':'nearest_big_city'}, inplace = True)


    cases_long = cases_long.merge(cases_long_to_join, 
                                  how = 'left', 
                                  on = ['time', 'nearest_big_city'])
    cases_long = cases_long.drop('nearest_big_city', axis = 1)



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
    #drop_times = cases.time[0:(k + t_lag)]
    drop_times = cases.time[0:t_lag]
    cases_long = cases_long[~cases_long['time'].isin(drop_times)]



    if write_to_file:
        cases_long.to_parquet("../../output/data/basic_nn/prefit_train_test/k" + str(k) + ".gzip", compression = "gzip")
        cases_transform_output.to_parquet("../../output/data/basic_nn/prefit_train_test/k" + str(k) + "_cases_transform_output.gzip", compression = "gzip")
        #cases_long_train.to_parquet("../../output/data/basic_nn/prefit_train_test/k" + str(k) + "_train.gzip", compression = "gzip")
        #cases_long_test.to_parquet("../../output/data/basic_nn/prefit_train_test/k" + str(k) + "_test.gzip", compression = "gzip")

    else:
        return cases_long, cases_transform_output

def get_train_test(dat, wrt, test_size):
    #split train/test
    dat_train = dat[dat[wrt] <= dat[wrt].quantile(1 - test_size)]
    dat_test = dat[dat[wrt] > dat[wrt].quantile(1 - test_size)]
    return dat_train, dat_test
    







if __name__ == "__main__":
    cases_train, cases_test = create_measles_data(k = 52, 
                                                  t_lag = 130, 
                                                  susc_data_loc = "../../../data/tsir_susceptibles/tsir_susceptibles.csv", 
                                                  birth_data_loc = "../../../data/births/ewBu4464.csv",
                                                  write_to_file = False,
                                                  verbose = True)

    cases_train, cases_test = get_train_test(dat = cases, wrt = 'time', test_size = 0.3)


    k_to_not_run = []
    #k_seq = np.concatenate((np.arange(1, 13, 1), np.arange(12, 53, 4)))
    k_seq = [51, 52]
    k_to_run = [k for k in k_seq if not k in k_to_not_run]



    for i in k_to_run:
        create_measles_data(k = i, t_lag = 130)
        print("#############" + str(i) + " written to file#############")


    #print column names
    for i in cases_long.columns:
        print(i)
