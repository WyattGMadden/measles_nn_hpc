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
    cases, transform_data = mdl.create_measles_data(
        k=config['k'], 
        t_lag=130, 
        cases_data_loc=config['cases_data_loc'], 
        pop_data_loc=config['pop_data_loc'], 
        coords_data_loc=config['coords_data_loc'], 
        susc_data_loc=config['susc_data_loc'], 
        birth_data_loc=config['birth_data_loc'], 
        top_12_cities=config['top_12_cities'], 
        verbose=config['verbose']
    )
    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, config['test_size'])
    
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
    parser.add_argument('--num-samples', type=int, default=100, help='Number of tuning samples')
    parser.add_argument('--max-num-epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--gpus-per-trial', type=float, default=1, help='GPUs per trial')
    args = parser.parse_args()

    # Configuration for hyperparameter tuning
    config = {
        "k": args.k,
        "cases_data_loc": os.path.abspath("../../../data/data_from_measles_competing_risks/inferred_cases_urban.csv"),
        "pop_data_loc": os.path.abspath("../../../data/data_from_measles_competing_risks/inferred_pop_urban.csv"),
        "coords_data_loc": os.path.abspath("../../../data/data_from_measles_competing_risks/coordinates_urban.csv"),
        "susc_data_loc": os.path.abspath("../../../output/data/tsir_susceptibles/tsir_susceptibles.csv"),
        "birth_data_loc": os.path.abspath("../../../data/data_from_measles_competing_risks/ewBu4464.csv"),
        "top_12_cities": True,
        "verbose": True,
        "test_size": 0.251197,
        "num_epochs": args.max_num_epochs,
        "lr": tune.uniform(0.0001, 0.1),  # Using random search for learning rate
        "hidden_dim": tune.grid_search([240, 480, 721, 961, 1201]),
        "weight_decay": tune.uniform(0.00001, 0.1),  # Using random search for weight decay
        "num_hidden_layers": tune.grid_search([1, 2, 3, 4])  # Adding this line to include variable hidden layers
    }

    # Start Ray Tune
    result = tune.run(
        train_with_tuning,
        resources_per_trial={"cpu": 1, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples
    )

    best_trial = result.get_best_trial("test_mse", "min", "last")

    # Create a summary DataFrame
    trials_data = []
    for trial in result.trials:
        trial_data = {
            "trial_id": trial.trial_id,
            "test_mse": trial.last_result["test_mse"],
            "is_best": trial.trial_id == best_trial.trial_id
        }
        # Unpack the config dictionary into individual columns
        trial_data.update(trial.config)
        trials_data.append(trial_data)

    df = pd.DataFrame(trials_data)
    df.sort_values("test_mse", inplace=True)

    # Save to CSV
    save_dir = "../../../output/figures/basic_nn/raytune_hp_optim/"
    df.to_csv(save_dir + "raytune_hp_optim_"+ "k_" + str(args.k) + ".csv", index=False)

    print("Results saved to hyperparameter_optimization_results.csv")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final test MSE: {}".format(best_trial.last_result["test_mse"]))

if __name__ == '__main__':
    main()
