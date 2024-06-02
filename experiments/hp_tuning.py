import os
import json

import astra
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour, plot_edf, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice

from sinogram.sinogram_generator import Sinogram
from noise2inverse.n2i import N2I
####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Phantom
PHANTOM_NAME = 'foam_phantom.npy'
PHANTOM_PATH = 'phantoms/save/'

# Sinogram
N_PROJECTIONS = 1024
N_ITERATIONS = 200
ATTENUATION = 5e-3
PHOTON_COUNT = 10
K = 4

REC_ALGORITHM = 'FBP_CUDA'

# Training hyperparameters
EPS = 10
BS = 8
LR = 0.005

# Optuna
N_TRIALS = 3

####################################################################################################
#                                           FUNCTIONS                                              #
####################################################################################################

def save_plot(function, name):
    fig = function
    os.makedirs("optuna_results", exist_ok=True)
    fig.write_image(f"optuna_results/{name}.png", scale=3)


def generate_plots(study):
    importance_params = optuna.importance.get_param_importances(study)
    print("Importance hyperparameters", importance_params)
    keys_important = list(importance_params.keys())
    save_plot(plot_optimization_history(study), "optimization_history")
    save_plot(plot_param_importances(study), "param_importances")
    save_plot(plot_slice(study, params=keys_important[:4]), "slice")
    save_plot(plot_parallel_coordinate(study, params=keys_important[:4]), "parallel_coordinate")
    save_plot(plot_edf(study), "edf")
    save_plot(plot_contour(study, params=keys_important[:2]), "contour")


def objective(trial):
    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    k = trial.suggest_int("k", 3, 5)
    strategy = trial.suggest_categorical("strategy", ["X:1", "1:X"])

    # Load phantom
    foam = np.load(PHANTOM_PATH + PHANTOM_NAME)

    # Generate sinogram
    sinogram = Sinogram(foam, N_PROJECTIONS, N_ITERATIONS)
    sinogram.generate()
    sinogram.add_poisson_noise(attenuation=ATTENUATION, photon_count=PHOTON_COUNT)

    # Split data in K parts and reconstruct each split
    sinogram.split_data(K)
    rec_splits = sinogram.reconstruct_splits(sinogram.split_sinograms, REC_ALGORITHM)

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n2i = N2I(foam, "unet", device, k, strategy, lr, BS, EPS)
    n2i.Train(rec_splits, original_image=foam)

    # Evaluate model
    denoised_phantom = n2i.Evaluate(rec_splits)

    # Convert denoised_phantom to numpy array if it's a PyTorch tensor
    denoised_phantom = denoised_phantom.cpu().numpy()
    loss_eval = torch.nn.functional.mse_loss(denoised_phantom, foam)

    # We will need to maximize based on the evaluate MSD
    return loss_eval


####################################################################################################
#                                              MAIN                                                #
####################################################################################################

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics:")
    print("   Number of finished trials: ", len(study.trials))
    print("   Number of pruned trials: ", len(pruned_trials))
    print("   Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_hyperparameters = {"value": trial.value, "params": trial.params}

    os.makedirs("optuna_results", exist_ok=True)
    with open('optuna_results/best_hyperopt.json', 'w') as fp:
        json.dump(best_hyperparameters, fp)

    # Save all trials information
    study.trials_dataframe().to_csv("optuna_results/hyperopt_data.csv")

    # Save plots
    generate_plots(study)