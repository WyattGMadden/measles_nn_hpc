import argparse
import sys
import os
import numpy as np
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
    cases, transform_data = mdl.create_measles_data(k=config['k'], t_lag=130, susc_data_loc=config['susc_data_loc'], birth_data_loc=config['birth_data_loc'], top_12_cities=config['top_12_cities'], verbose=config['verbose'])
    train_data, test_data, num_features, id_train, id_test = fbf.process_data(cases, config['test_size'])
    
    # Define model
    model = fbf.NeuralNetwork(num_features, config['hidden_dim'], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(config['num_epochs']):
        fbf.train(model=model, device=device, train_loader=DataLoader(train_data, batch_size=64, shuffle=True), optimizer=optimizer, loss_fn=loss_fn, epoch=epoch, log_interval=100)
        loss = fbf.test(model, device, DataLoader(test_data, batch_size=64, shuffle=False), loss_fn)

    # Save train loss
    pred_train = model(train_data.X.to(device)).to("cpu").detach().numpy()
    train_mse = np.mean((pred_train - train_data.y.detach().numpy())**2)
    train.report({'train_mse': train_mse})  # Use tune.report to send metrics

    # Save test loss
    pred_test = model(test_data.X.to(device)).to("cpu").detach().numpy()
    test_mse = np.mean((pred_test - test_data.y.detach().numpy())**2)
    train.report({'test_mse': test_mse})  # Use tune.report to send metrics

    

def main():
    parser = argparse.ArgumentParser(description='Tune Neural Network Hyperparameters')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of tuning samples')
    parser.add_argument('--max-num-epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--gpus-per-trial', type=float, default=1, help='GPUs per trial')
    args = parser.parse_args()

    # Configuration for hyperparameter tuning
    config = {
        "k": 52,
        "susc_data_loc": os.path.abspath("../../../data/tsir_susceptibles/tsir_susceptibles.csv"),
        "birth_data_loc": os.path.abspath("../../../data/births/ewBu4464.csv"),
        "top_12_cities": True,
        "verbose": True,
        "test_size": 0.3,
        "num_epochs": args.max_num_epochs,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        # 1/6 input dim - 5/6 input dim
        "hidden_dim": tune.grid_search([240, 480, 721, 961, 1201]),
        "weight_decay": tune.grid_search([0.01, 0.005, 0.001, 0.0005, 0.0001])
    }

    # Start Ray Tune
    result = tune.run(
        train_with_tuning,
        resources_per_trial={"cpu": 1, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples
    )

    best_trial = result.get_best_trial("test_mse", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["test_mse"]))

if __name__ == '__main__':
    main()


