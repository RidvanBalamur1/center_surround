import torch
import torch.nn as nn
import optuna
from typing import Dict, Any, Type
from center_surround.models import KlindtCoreReadout2D
from center_surround.models.klindtSurround import (
    KlindtCoreReadoutPerChannel2D,
    KlindtCoreReadoutNMasks2D,
    KlindtCoreReadoutONOFF2D,
    KlindtCoreReadoutDedicatedONOFFMixed2D,
)
from center_surround.utils import correlation


def train_and_evaluate(model, dataloaders, num_epochs, lr, device, patience=10):
    """Train model with early stopping and return validation correlation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    best_val_corr = -float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        for images, responses in dataloaders['train']:
            images, responses = images.to(device), responses.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, responses) + model.regularizer()
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, responses in dataloaders['validation']:
                images = images.to(device)
                output = model(images)
                all_preds.append(output.cpu())
                all_targets.append(responses)
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        val_corr = correlation(preds, targets).mean().item()
        
        # Early stopping check
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_corr


def create_objective(dataloaders, input_size, in_channels, num_neurons, device, num_epochs=30):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Hyperparameters to tune
        # kernel_size = trial.suggest_int('kernel_size', 13, 31, step=4)
        smoothness_reg = trial.suggest_float('smoothness_reg', 1e-6, 1e-1, log=True)
        #center_mass_reg = trial.suggest_float('center_mass_reg', 0.0, 1.5)
        weights_reg = trial.suggest_float('weights_reg', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        init_kernels_sigma = trial.suggest_float('init_kernels_sigma', 0.1, 0.5)
        
        # Create model
        model = KlindtCoreReadout2D(
            image_size=input_size,
            image_channels=in_channels,
            kernel_sizes=[24,24],
            num_kernels=[4],
            act_fns=['relu'],
            init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
            init_kernels=f"gaussian:{init_kernels_sigma}",
            num_neurons=num_neurons,
            smoothness_reg=smoothness_reg,
            sparsity_reg=0,
            center_mass_reg= 0,#center_mass_reg,
            mask_reg=0.0001,
            weights_reg= weights_reg,
            dropout_rate=0.2,
            batch_norm=True,
            bn_cent=False,
            kernel_constraint="norm",
            weights_constraint="abs",
            mask_constraint="abs",
            final_relu=True,
            seed=42,
        )
        
        # Train and evaluate with early stopping
        val_corr = train_and_evaluate(model, dataloaders, num_epochs, learning_rate, device, patience=10)
        
        return val_corr
    
    return objective


def run_hyperparameter_search(dataloaders, input_size, in_channels, num_neurons,
                               device, n_trials=30, num_epochs=30):
    """Run Optuna hyperparameter search."""

    objective = create_objective(dataloaders, input_size, in_channels, num_neurons, device, num_epochs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'study': study,
    }


def create_objective_surround(dataloaders, input_size, in_channels, num_neurons, device, num_epochs=30, num_masks=2):
    """Create Optuna objective function for KlindtCoreReadoutNMasks2D (surround model with N masks)."""

    def objective(trial):
        # Hyperparameters to tune
        smoothness_reg = trial.suggest_float('smoothness_reg', 1e-6, 1e-1, log=True)
        sparsity_reg = trial.suggest_float('sparsity_reg', 1e-6, 1e-1, log=True)
        weights_reg = trial.suggest_float('weights_reg', 1e-5, 1e-1, log=True)
        mask_reg = trial.suggest_float('mask_reg', 1e-4, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        init_kernels_sigma = trial.suggest_float('init_kernels_sigma', 0.1, 0.5)

        # Create model with N spatial masks (default 2 for center/surround)
        model = KlindtCoreReadoutNMasks2D(
            image_size=input_size,
            image_channels=in_channels,
            kernel_sizes=[24, 24],
            num_kernels=[4],
            act_fns=['relu'],
            init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
            init_kernels=f"gaussian:{init_kernels_sigma}",
            num_neurons=num_neurons,
            num_masks=num_masks,  # 2 masks: center and surround
            smoothness_reg=smoothness_reg,
            sparsity_reg=sparsity_reg,
            center_mass_reg=0,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            dropout_rate=0.2,
            batch_norm=True,
            bn_cent=False,
            kernel_constraint="norm",
            weights_constraint="abs",
            mask_constraint="abs",
            final_relu=True,
            seed=42,
        )

        # Train and evaluate with early stopping
        val_corr = train_and_evaluate(model, dataloaders, num_epochs, learning_rate, device, patience=10)

        return val_corr

    return objective


def run_hyperparameter_search_surround(dataloaders, input_size, in_channels, num_neurons,
                                        device, n_trials=30, num_epochs=30, num_masks=2):
    """Run Optuna hyperparameter search for KlindtCoreReadoutNMasks2D (surround model with N masks)."""

    objective = create_objective_surround(dataloaders, input_size, in_channels, num_neurons, device, num_epochs, num_masks)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'study': study,
    }


def create_objective_onoff(dataloaders, input_size, in_channels, num_neurons, device, num_epochs=30, n_on_kernels=2, n_off_kernels=2):
    """Create Optuna objective function for KlindtCoreReadoutONOFF2D (ON/OFF polarity model)."""

    def objective(trial):
        # Hyperparameters to tune
        smoothness_reg = trial.suggest_float('smoothness_reg', 1e-6, 1e-1, log=True)
        # sparsity_reg = trial.suggest_float('sparsity_reg', 1e-6, 1e-1, log=True)
        weights_reg = trial.suggest_float('weights_reg', 1e-5, 1e-1, log=True)
        mask_reg = trial.suggest_float('mask_reg', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        # init_kernels_sigma = trial.suggest_float('init_kernels_sigma', 0.1, 0.5)

        # Create model with ON/OFF polarity
        model = KlindtCoreReadoutONOFF2D(
            image_size=input_size,
            image_channels=in_channels,
            kernel_sizes=[24, 24],
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            act_fns=['relu'],
            init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
            init_kernels=f"gaussian:0.28",
            num_neurons=num_neurons,
            smoothness_reg=smoothness_reg,
            sparsity_reg=0,
            center_mass_reg=0.2,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            dropout_rate=0.2,
            batch_norm=True,
            bn_cent=False,
            kernel_constraint="norm",
            weights_constraint="abs",
            mask_constraint="abs",
            final_relu=True,
            seed=42,
        )

        # Train and evaluate with early stopping
        val_corr = train_and_evaluate(model, dataloaders, num_epochs, learning_rate, device, patience=10)

        return val_corr

    return objective


def run_hyperparameter_search_onoff(dataloaders, input_size, in_channels, num_neurons,
                                     device, n_trials=30, num_epochs=30, n_on_kernels=2, n_off_kernels=2):
    """Run Optuna hyperparameter search for KlindtCoreReadoutONOFF2D (ON/OFF polarity model)."""

    objective = create_objective_onoff(dataloaders, input_size, in_channels, num_neurons, device, num_epochs, n_on_kernels, n_off_kernels)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'study': study,
    }


def create_objective_dedicated_onoff_mixed(dataloaders, input_size, in_channels, num_neurons, device,
                                            num_epochs=30, n_on_kernels=2, n_off_kernels=2, n_mixed_kernels=2):
    """Create Optuna objective function for KlindtCoreReadoutDedicatedONOFFMixed2D.

    This model has 6 dedicated kernels:
    - 2 ON kernels (positive weights) → dedicated to ON mask
    - 2 OFF kernels (negative weights) → dedicated to OFF mask
    - 2 Mixed kernels (1 ON-like, 1 OFF-like) → dedicated to Mixed mask
    """

    def objective(trial):
        # Hyperparameters to tune
        smoothness_reg = trial.suggest_float('smoothness_reg', 1e-6, 1e-1, log=True)
        weights_reg = trial.suggest_float('weights_reg', 1e-5, 1e-1, log=True)
        mask_reg = trial.suggest_float('mask_reg', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)

        # Create model with dedicated ON/OFF/Mixed kernels
        model = KlindtCoreReadoutDedicatedONOFFMixed2D(
            image_size=input_size,
            image_channels=in_channels,
            kernel_sizes=[24, 24],
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            n_mixed_kernels=n_mixed_kernels,
            act_fns=['relu'],
            init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
            init_kernels="gaussian:0.28",
            num_neurons=num_neurons,
            smoothness_reg=smoothness_reg,
            sparsity_reg=0,
            center_mass_reg=0,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            dropout_rate=0.2,
            batch_norm=True,
            bn_cent=False,
            kernel_constraint="norm",
            weights_constraint="abs",
            mask_constraint="abs",
            final_relu=True,
            seed=42,
        )

        # Train and evaluate with early stopping
        val_corr = train_and_evaluate(model, dataloaders, num_epochs, learning_rate, device, patience=10)

        return val_corr

    return objective


def run_hyperparameter_search_dedicated_onoff_mixed(dataloaders, input_size, in_channels, num_neurons,
                                                     device, n_trials=30, num_epochs=30,
                                                     n_on_kernels=2, n_off_kernels=2, n_mixed_kernels=2):
    """Run Optuna hyperparameter search for KlindtCoreReadoutDedicatedONOFFMixed2D.

    This model has 6 dedicated kernels (2 ON + 2 OFF + 2 Mixed), each pathway
    with its own dedicated kernels to prevent conflicting optimization pressures.
    """

    objective = create_objective_dedicated_onoff_mixed(
        dataloaders, input_size, in_channels, num_neurons, device, num_epochs,
        n_on_kernels, n_off_kernels, n_mixed_kernels
    )

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'study': study,
    }
