import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.models import KlindtCoreReadout2D
from center_surround.training import train
from center_surround.utils import plot_kernels, plot_spatial_masks
from center_surround.utils import (
    compute_metrics, 
    plot_predictions_grid, 
    plot_correlation_vs_reliability, 
    bootstrap_reliability
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load best hyperparameters
with open('/home/ridvan/Documents/center_surround/outputs/hyperparam_results.pkl', 'rb') as f:
    results = pickle.load(f)

best_params = results['best_params']
print("Best hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Load data
data_path = "/home/ridvan/Documents/center_surround/data/exp_13_data.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32)

# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"\nInput: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Create model with best parameters
model = KlindtCoreReadout2D(
    image_size=input_size,
    image_channels=in_channels,
    kernel_sizes=[13,13],
    num_kernels=[4],
    act_fns=['relu'],
    init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    init_kernels=f"gaussian:0.2",
    num_neurons=num_neurons,
    smoothness_reg=best_params['smoothness_reg'],
    sparsity_reg=0.0,
    center_mass_reg=0.4,
    mask_reg=0.0001,
    weights_reg=0.001,
    dropout_rate=0.2,
    batch_norm=True,
    bn_cent=False,
    kernel_constraint="norm",
    weights_constraint="abs",
    mask_constraint="abs",
    final_relu=True,
    seed=42,
)

# Train with more epochs
print("\nTraining final model...")
model = train(model, dataloaders, num_epochs=100, lr=0.01, device=device)

# Evaluate
print("\nEvaluating...")
metrics = compute_metrics(model, dataloaders['test'], device)
print(f"Test MSE: {metrics['mse']:.4f}")
print(f"Mean Correlation: {metrics['mean_correlation']:.4f}")
print(f"Per-neuron Correlation: {metrics['correlation_per_neuron']}")

# Get predictions for plotting
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for images_batch, responses_batch in dataloaders['test']:
        images_batch = images_batch.to(device)
        output = model(images_batch)
        all_preds.append(output.cpu().numpy())
        all_targets.append(responses_batch.numpy())

predictions = np.concatenate(all_preds, axis=0)
targets = np.concatenate(all_targets, axis=0)
correlations = metrics['correlation_per_neuron']

# Plot predictions
print("\nPlotting predictions...")
fig1 = plot_predictions_grid(predictions, targets, correlations)
plt.show()

# Plot correlation vs reliability
if 'responses_test_by_trial' in raw_data:
    responses_by_trial = raw_data['responses_test_by_trial']
    print(f"Trial data shape: {responses_by_trial.shape}")
    
    if responses_by_trial.shape[0] == 30 and responses_by_trial.shape[2] == 30:
        pass
    elif responses_by_trial.shape[0] == 30 and responses_by_trial.shape[1] == 30:
        responses_by_trial = responses_by_trial.transpose(1, 2, 0)
    
    reliability, _ = bootstrap_reliability(responses_by_trial)
    print(f"Reliability: {reliability}")
    
    fig2 = plot_correlation_vs_reliability(correlations, reliability)
    plt.show()

# Plot kernels
print("\nPlotting kernels...")
fig3 = plot_kernels(model)
plt.show()

# Plot spatial masks
print("\nPlotting spatial masks...")
fig4 = plot_spatial_masks(model)
plt.show()

# Save the trained model
model_path = '/home/ridvan/Documents/center_surround/outputs/best_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
