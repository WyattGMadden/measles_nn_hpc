import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from ray import train
import ray

# Adjust Python path for the script
original_sys_path = sys.path.copy()
data_processing_path = os.path.abspath('../data_processing/')
sys.path.append(data_processing_path)

import prevac_measles_data_loader as mdl
sys.path = original_sys_path

import full_basic_functions as fbf

# Initialize Ray and ensure worker nodes have the correct Python path
ray.init(runtime_env={"env_vars": {"PYTHONPATH": str(data_processing_path)}})

def train_with_tuning(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and process data
    full_cases_loc = config['cases_data_loc'] + "/k" + str(config['k']) + "_tlag" + str(config['t_lag']) + ".gzip"
    cases = pd.read_parquet(full_cases_loc)

    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, config['year_test_cutoff'])
    
    # Define model with the specified number of hidden layers
    model = fbf.NeuralNetwork(num_features, config['hidden_dim'], 1, num_hidden_layers=config['num_hidden_layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(config['num_epochs']):
        fbf.train(model=model, device=device, train_loader=DataLoader(train_data, batch_size=64, shuffle=True), optimizer=optimizer, loss_fn=loss_fn, epoch=epoch, log_interval=100)
        loss = fbf.test(model, device, DataLoader(test_data, batch_size=64, shuffle=False), loss_fn)

    # Reporting metrics
    pred_train = model(train_data.X.to(device)).to("cpu").detach().numpy()
    train_mse = np.mean((pred_train - train_data.y.detach().numpy())**2)
    train.report({'train_mse': train_mse})
    pred_test = model(test_data.X.to(device)).to("cpu").detach().numpy()
    test_mse = np.mean((pred_test - test_data.y.detach().numpy())**2)
    train.report({'test_mse': test_mse})



def main():
    parser = argparse.ArgumentParser(description='Tune Neural Network Hyperparameters')
    parser.add_argument('--k', type=int, default=1, help='k-steps ahead')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of tuning samples')
    parser.add_argument('--max-num-epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--gpus-per-trial', type=float, default=1, help='GPUs per trial')
    args = parser.parse_args()

    # Adjust t_lag options based on the value of k
    if args.k < 26:
        t_lag_options = [26, 26*2, 26*3]
    elif args.k >= 26 and args.k < 52:
        t_lag_options = [26*2, 26*3, 26*4]
    else:
        t_lag_options = [26*3, 26*4, 26*5]

    # Configuration for hyperparameter tuning
    config = {
        "k": args.k,
        "cases_data_loc": os.path.abspath("../../../output/data/prefit_cases"),
        "top_12_cities": False,
        "verbose": True,
        "year_test_cutoff": 61,
        "num_epochs": args.max_num_epochs,
        "lr": 0.001,  # Using random search for learning rate
        "t_lag": tune.grid_search(t_lag_options),
        "hidden_dim": tune.grid_search([240, 721, 1201]),
        "weight_decay": tune.uniform(0.0001, 0.1),  # Using random search for weight decay
        "num_hidden_layers": tune.grid_search([1, 2, 3])
    }

    # Start Ray Tune
    result = tune.run(
        train_with_tuning,
        resources_per_trial={"cpu": 12, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples
    )

    # Process results
    best_trial = result.get_best_trial("test_mse", "min", "last")
    trials_data = [{**{"trial_id": trial.trial_id, "test_mse": trial.last_result["test_mse"], "is_best": trial.trial_id == best_trial.trial_id}, **trial.config} for trial in result.trials]
    df = pd.DataFrame(trials_data)
    df.sort_values("test_mse", inplace=True)
    save_dir = "../../../output/figures/basic_nn/raytune_hp_optim/"
    df.to_csv(save_dir + f"raytune_hp_optim_k_{args.k}.csv", index=False)
    print("Results saved to hyperparameter_optimization_results.csv")
    print("Best trial config:", best_trial.config)
    print("Best trial final test MSE:", best_trial.last_result["test_mse"])

if __name__ == '__main__':
    main()
