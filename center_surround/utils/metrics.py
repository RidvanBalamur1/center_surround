import torch
import numpy as np


def correlation(predictions, targets):
    """Compute Pearson correlation per neuron."""
    pred = predictions - predictions.mean(dim=0)
    targ = targets - targets.mean(dim=0)
    
    num = (pred * targ).sum(dim=0)
    den = torch.sqrt((pred ** 2).sum(dim=0) * (targ ** 2).sum(dim=0))
    
    return num / (den + 1e-8)


def compute_metrics(model, dataloader, device):
    """Compute evaluation metrics on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, responses in dataloader:
            images = images.to(device)
            output = model(images)
            all_preds.append(output.cpu())
            all_targets.append(responses)
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    corr = correlation(preds, targets)
    mse = ((preds - targets) ** 2).mean().item()
    
    return {
        "mse": mse,
        "correlation_per_neuron": corr.numpy(),
        "mean_correlation": corr.mean().item(),
    }
