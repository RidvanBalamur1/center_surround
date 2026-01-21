import torch
import torch.nn as nn
import optuna
from typing import Dict, Any, Type
from center_surround.models import KlindtCoreReadout2D
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
        center_mass_reg = trial.suggest_float('center_mass_reg', 0.0, 1.5)
        weights_reg = trial.suggest_float('weights_reg', 1e-5, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        
        # Create model
        model = KlindtCoreReadout2D(
            image_size=input_size,
            image_channels=in_channels,
            kernel_sizes=[31,31],
            num_kernels=[4],
            act_fns=['relu'],
            init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
            init_kernels=f"gaussian:0.2",
            num_neurons=num_neurons,
            smoothness_reg=smoothness_reg,
            sparsity_reg=0,
            center_mass_reg=center_mass_reg,
            mask_reg=0.0001,
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
