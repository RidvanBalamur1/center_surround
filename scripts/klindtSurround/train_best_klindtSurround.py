import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PIL import Image
import cv2

from center_surround.data import load_raw_data, create_dataloaders
from center_surround.models.klindtSurround import KlindtCoreReadoutNMasks2D
from center_surround.training import train
from center_surround.utils import plot_kernels, plot_spatial_masks_n
from center_surround.utils import compute_lsta, plot_lsta_comparison_per_cell
from center_surround.utils import (
    compute_metrics,
    plot_predictions_grid,
    plot_correlation_vs_reliability,
    bootstrap_reliability,
    plot_correlation_distribution,
    plot_reliability_distribution,
    plot_fraction_of_ceiling,
    plot_example_predictions,
    plot_grid_predictions,
)

# Model name
model_name = "klindtSurround"

# Data path
data_path_file = "exp_13_data_4"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load best hyperparameters from surround model search
hyperparam_path = f'/home/ridvan/Documents/center_surround/outputs/{data_path_file}/{model_name}/hyperparam_results_latest.pkl'
with open(hyperparam_path, 'rb') as f:
    results = pickle.load(f)

best_params = results['best_params']
print(f"Loaded hyperparameters from: {hyperparam_path}")
print("Best hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Load data
data_path = f"/home/ridvan/Documents/center_surround/data/{data_path_file}.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32, normalize_images=True)

# Create output directory with timestamp at the start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'/home/ridvan/Documents/center_surround/outputs/{data_path_file}/{model_name}/run_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"\nInput: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Cell IDs used in your model (must match order of neurons in data)
# picked_cells = [187, 478, 684, 721, 723, 1036, 1091, 1260, 1443, 1506] Data set 1,2
picked_cells = [23, 27, 63, 75, 89, 90, 122, 129, 143, 148, 169, 212, 221, 312, 402, 441, 478, 500,
                507, 542, 582, 614, 676, 684, 721, 723, 755, 761, 807, 813, 881, 964,
                1054, 1056, 1059, 1091, 1122, 1145, 1156, 1189, 1200, 1262, 1278, 1317, 1429,
                1437, 1441, 1443, 1487, 1538, 1554, 1587, 1593, 1598, 1662, 1705]


# Model configuration - using tuned hyperparameters from search
# Using 2 spatial masks (center/surround) with 4 kernel channels
model_config = {
    'image_size': input_size,
    'image_channels': in_channels,
    'kernel_sizes': [24, 24],
    'num_kernels': [4],
    'num_masks': 2,  # 2 spatial masks: center and surround
    'act_fns': ['relu'],
    'init_scales': [[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    'init_kernels': f"gaussian:{best_params['init_kernels_sigma']}",
    'num_neurons': num_neurons,
    'smoothness_reg': best_params['smoothness_reg'],
    'sparsity_reg': best_params['sparsity_reg'],
    'center_mass_reg': 0,
    'mask_reg': best_params['mask_reg'],
    'weights_reg': best_params['weights_reg'],
    'dropout_rate': 0.2,
    'batch_norm': True,
    'bn_cent': False,
    'kernel_constraint': 'norm',
    'weights_constraint': 'abs',
    'mask_constraint': 'abs',
    'final_relu': True,
    'seed': 42,
}

# Save config at the start of the run
config = {
    'data_path': data_path,
    'picked_cells': picked_cells,
    'best_params': best_params,
    'model_config': model_config,
    'batch_size': 32,
    'num_epochs': 100,
    'device': str(device),
}
with open(f'{output_dir}/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print(f"Config saved to {output_dir}/config.pkl")

# Create model with best parameters (2 spatial masks for center/surround)
model = KlindtCoreReadoutNMasks2D(
    image_size=model_config['image_size'],
    image_channels=model_config['image_channels'],
    kernel_sizes=model_config['kernel_sizes'],
    num_kernels=model_config['num_kernels'],
    act_fns=model_config['act_fns'],
    init_scales=model_config['init_scales'],
    init_kernels=model_config['init_kernels'],
    num_neurons=model_config['num_neurons'],
    num_masks=model_config['num_masks'],
    smoothness_reg=model_config['smoothness_reg'],
    sparsity_reg=model_config['sparsity_reg'],
    center_mass_reg=model_config['center_mass_reg'],
    mask_reg=model_config['mask_reg'],
    weights_reg=model_config['weights_reg'],
    dropout_rate=model_config['dropout_rate'],
    batch_norm=model_config['batch_norm'],
    bn_cent=model_config['bn_cent'],
    kernel_constraint=model_config['kernel_constraint'],
    weights_constraint=model_config['weights_constraint'],
    mask_constraint=model_config['mask_constraint'],
    final_relu=model_config['final_relu'],
    seed=model_config['seed'],
)

# Train with more epochs using tuned learning rate
print("\nTraining final model...")
model = train(model, dataloaders, num_epochs=100, lr=best_params['learning_rate'], device=device, early_stopping=False)

# Check validation correlation first
print("\nValidation set evaluation:")
val_metrics = compute_metrics(model, dataloaders['validation'], device)
print(f"Val Mean Correlation: {val_metrics['mean_correlation']:.4f}")

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
fig1.savefig(f'{output_dir}/predictions.png', dpi=150, bbox_inches='tight')
# plt.show()

# Plot correlation distribution
print("\nPlotting correlation distribution...")
fig_corr_dist = plot_correlation_distribution(correlations)
fig_corr_dist.savefig(f'{output_dir}/correlation_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig_corr_dist)

# Plot example predictions (actual vs predicted scatter)
print("\nPlotting example predictions...")
fig_example = plot_example_predictions(predictions, targets, correlations)
fig_example.savefig(f'{output_dir}/example_predictions.png', dpi=150, bbox_inches='tight')
plt.close(fig_example)

# Plot grid predictions (all neurons)
print("\nPlotting grid predictions...")
grid_preds_dir = f'{output_dir}/grid_predictions'
os.makedirs(grid_preds_dir, exist_ok=True)
grid_figs = plot_grid_predictions(predictions, targets, correlations)
for i, fig in enumerate(grid_figs):
    fig.savefig(f'{grid_preds_dir}/grid_predictions_{i}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# Plot correlation vs reliability (with reliability-dependent plots)
if 'responses_test_by_trial' in raw_data:
    responses_by_trial = raw_data['responses_test_by_trial']
    print(f"Trial data shape: {responses_by_trial.shape}")

    if responses_by_trial.shape[0] == 30 and responses_by_trial.shape[2] == 30:
        pass
    elif responses_by_trial.shape[0] == 30 and responses_by_trial.shape[1] == 30:
        responses_by_trial = responses_by_trial.transpose(1, 2, 0)

    reliability, _ = bootstrap_reliability(responses_by_trial)
    print(f"Reliability: {reliability}")

    # Correlation vs Reliability (with trend line)
    fig2 = plot_correlation_vs_reliability(correlations, reliability)
    fig2.savefig(f'{output_dir}/correlation_vs_reliability.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Reliability distribution
    print("\nPlotting reliability distribution...")
    fig_rel_dist = plot_reliability_distribution(reliability)
    fig_rel_dist.savefig(f'{output_dir}/reliability_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig_rel_dist)

    # Fraction of ceiling
    foc = correlations / reliability
    print("\nPlotting fraction of ceiling...")
    fig_foc = plot_fraction_of_ceiling(foc)
    fig_foc.savefig(f'{output_dir}/fraction_of_ceiling.png', dpi=150, bbox_inches='tight')
    plt.close(fig_foc)

    # Update example predictions with reliability and FoC
    print("\nPlotting example predictions with reliability...")
    fig_example_rel = plot_example_predictions(predictions, targets, correlations, reliability, foc)
    fig_example_rel.savefig(f'{output_dir}/example_predictions_with_reliability.png', dpi=150, bbox_inches='tight')
    plt.close(fig_example_rel)

# Plot kernels
print("\nPlotting kernels...")
fig3 = plot_kernels(model)
fig3.savefig(f'{output_dir}/kernels.png', dpi=150, bbox_inches='tight')
# plt.show()

# Plot spatial masks (N masks per neuron)
print("\nPlotting spatial masks...")
spatial_masks_dir = f'{output_dir}/spatial_masks'
os.makedirs(spatial_masks_dir, exist_ok=True)
spatial_mask_figs = plot_spatial_masks_n(model, neurons_per_plot=8)
for i, fig in enumerate(spatial_mask_figs):
    fig.savefig(f'{spatial_masks_dir}/spatial_masks_{i}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# Compute LSTA on NATURAL IMAGES (not test images)
print("\nComputing LSTA...")

def center_crop(image, crop_size=432):
    """Center crop an image to crop_size x crop_size."""
    h, w = image.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top:top+crop_size, left:left+crop_size]

img_path = "/home/ridvan/Documents/exp/13_exp/20251208/stims/images"
img_numbers = [1, 3, 12, 620]  # The 4 natural images used in exp
dmd_size = 432  # DMD size used in experiment

lsta_images = []
for img_num in img_numbers:
    filename = f"image_{img_num}_image.png"
    image = Image.open(os.path.join(img_path, filename)).convert("L")
    img_data = np.array(image, dtype=np.float32) / 255.0

    # Center crop to DMD size (432x432)
    img_data = center_crop(img_data, dmd_size)

    # Normalize same way as training data
    img_data = (img_data - img_data.mean()) * 5

    # Resize to model input size (108x108 or whatever your model expects)
    img_data = cv2.resize(img_data, (input_size, input_size), interpolation=cv2.INTER_AREA)
    img_data = img_data[np.newaxis, :, :]  # Add channel dim: [1, H, W]
    lsta_images.append(img_data)

lsta_images = np.array(lsta_images)  # [4, 1, 108, 108]
    
# Load RF fits for ellipse overlay
rf_fits_path = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/Chirp_Checkerboard_Analysis_mix/sta_data_mix.pkl'
with open(rf_fits_path, 'rb') as f:
    rf_fits = pickle.load(f)

# Load experimental LSTA data
lsta_exp_path = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/768_pert/lSTAs_data_total_lstas.pkl'
with open(lsta_exp_path, 'rb') as f:
    all_lsta_exp = pickle.load(f)

# Extract experimental LSTA for your cells
lsta_exp = []
for cell_id in picked_cells:
    lsta_exp.append(all_lsta_exp[0][cell_id])
lsta_exp = np.array(lsta_exp)

# Compute model LSTA
lsta_model = compute_lsta(model, lsta_images, device=device)
print(f"Model LSTA shape: {lsta_model.shape}")
print(f"Exp LSTA shape: {lsta_exp.shape}")

# Save per-cell comparison plots WITH RF ellipse overlay
print("\nSaving LSTA comparison plots...")
saved_paths = plot_lsta_comparison_per_cell(
    images=lsta_images,
    lsta_model=lsta_model,
    lsta_exp=lsta_exp,
    output_dir=output_dir,
    rf_fits=rf_fits,           # RF fit data for ellipse overlay
    cell_ids=picked_cells,     # Cell IDs matching neurons
    rf_dim=72,                 # Original STA dimension
    cmap='RdBu_r',
    vmax_thresh=0.5,
    expon_treat=3,
    zoom=2.5,                  # Zoom around RF center (1.0 = no zoom)
    target_size=72,          # Size of the final LSTA image
)
print(f"Saved {len(saved_paths)} LSTA comparison plots to {output_dir}/lsta_per_cell/")

# Save the trained model
model_path = f'{output_dir}/best_model_1.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

# Save metrics and predictions
results_path = f'{output_dir}/results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump({
        'val_metrics': val_metrics,
        'test_metrics': metrics,
        'predictions': predictions,
        'targets': targets,
        'correlations': correlations,
    }, f)
print(f"Results saved to {results_path}")

print(f"\n{'='*50}")
print(f"All outputs saved to {output_dir}")
print(f"{'='*50}")
print(f"  - config.pkl (model config, hyperparameters, cell IDs)")
print(f"  - best_model.pth (trained model weights)")
print(f"  - results.pkl (metrics, predictions, correlations)")
print(f"  - predictions.png (response curves)")
print(f"  - correlation_distribution.png")
print(f"  - example_predictions.png (actual vs predicted scatter)")
print(f"  - grid_predictions_*.png (all neurons actual vs predicted)")
print(f"  - correlation_vs_reliability.png (with trend line)")
print(f"  - reliability_distribution.png")
print(f"  - fraction_of_ceiling.png")
print(f"  - example_predictions_with_reliability.png")
print(f"  - kernels.png")
print(f"  - spatial_masks.png")
print(f"  - lsta_per_cell/ (LSTA comparison plots)")



