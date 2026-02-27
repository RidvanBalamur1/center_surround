import torch
import torch.nn as nn
import numpy as np


def correlation_per_neuron(output, target, eps=1e-8):
    """Compute correlation between output and target per neuron."""
    delta_out = output - output.mean(dim=0, keepdim=True)
    delta_target = target - target.mean(dim=0, keepdim=True)

    var_out = delta_out.pow(2).mean(dim=0)
    var_target = delta_target.pow(2).mean(dim=0)

    corrs = (delta_out * delta_target).mean(dim=0) / (
        (var_out + eps).sqrt() * (var_target + eps).sqrt()
    )
    return corrs


def train_one_epoch(model, dataloader, optimizer, criterion, device, use_regularizer=True):
    model.train()
    total_loss = 0
    all_outputs = []
    all_targets = []

    for images, responses in dataloader:
        images, responses = images.to(device), responses.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, responses)

        # Add regularizer if model has one
        if use_regularizer and hasattr(model, 'regularizer'):
            loss = loss + model.regularizer()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_outputs.append(output.detach())
        all_targets.append(responses.detach())

    # Compute correlation for monitoring
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    corr = correlation_per_neuron(all_outputs, all_targets).mean().item()

    return total_loss / len(dataloader), corr


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, responses in dataloader:
            images, responses = images.to(device), responses.to(device)
            output = model(images)
            loss = criterion(output, responses)
            total_loss += loss.item()
            all_outputs.append(output)
            all_targets.append(responses)

    # Compute correlation for monitoring
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    corr = correlation_per_neuron(all_outputs, all_targets).mean().item()

    return total_loss / len(dataloader), corr


def train(model, dataloaders, num_epochs, lr, device, use_regularizer=True, patience=15,
          lr_decay_factor=0.3, scheduler_patience=5, min_lr=None, early_stopping=True):
    """
    Train model with AdamW optimizer and ReduceLROnPlateau scheduler.

    Matches OpenRetinaUI training setup:
    - AdamW optimizer (with weight decay)
    - ReduceLROnPlateau scheduler monitoring validation correlation
    - Early stopping based on validation loss

    Args:
        model: The model to train
        dataloaders: Dict with 'train' and 'validation' dataloaders
        num_epochs: Maximum number of epochs
        lr: Initial learning rate
        device: Device to train on
        use_regularizer: Whether to add model regularizer to loss
        patience: Early stopping patience (epochs without improvement)
        lr_decay_factor: Factor to reduce LR by (default: 0.3)
        scheduler_patience: Epochs to wait before reducing LR (default: 5)
        min_lr: Minimum learning rate (default: lr * decay_factor^3)
        early_stopping: Whether to use early stopping (default: True)
    """
    model = model.to(device)

    # AdamW optimizer (matches OpenRetinaUI)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Poisson loss (same as before)
    criterion = nn.PoissonNLLLoss(log_input=False)

    # Learning rate scheduler (matches OpenRetinaUI)
    if min_lr is None:
        min_lr = lr * (lr_decay_factor ** 3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize correlation
        factor=lr_decay_factor,
        patience=scheduler_patience,
        threshold=0.0005,
        threshold_mode='abs',
        min_lr=min_lr,
    )

    best_val_loss = float('inf')
    best_val_corr = -float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_corr = train_one_epoch(
            model, dataloaders['train'], optimizer, criterion, device, use_regularizer
        )
        val_loss, val_corr = evaluate(model, dataloaders['validation'], criterion, device)

        # Step scheduler based on validation correlation
        scheduler.step(val_corr)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Corr: {train_corr:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Corr: {val_corr:.4f}, "
              f"LR: {current_lr:.2e}")

        # Track best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_corr = val_corr
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val Loss: {best_val_loss:.4f}, Val Corr: {best_val_corr:.4f}")

    return model
