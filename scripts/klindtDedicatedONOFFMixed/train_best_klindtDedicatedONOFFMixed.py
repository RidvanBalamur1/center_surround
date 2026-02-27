import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PIL import Image
import cv2

from center_surround.data import load_raw_data, create_dataloaders
from center_surround.models.klindtSurround import KlindtCoreReadoutDedicatedONOFFMixed2D
from center_surround.training import train
from center_surround.utils import plot_kernels_dedicated
from center_surround.utils import (
    compute_lsta, compute_lsta_masked,
    create_batch_masks_for_conditions,
    plot_lsta_comparison_per_cell, plot_cell_id_cards,
)
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

EXP_NUM = 13
EXP_DATE = "20251208"

# Select which cell type to train on
# cell_type = "reliability_bigger_than_0point5" 
cell_type = "all_cells_11kernels" 

# Model name
model_name = "klindtDedicatedONOFFMixed"

# Data path
data_path_file = f"exp_{EXP_NUM}_full_data"

# Cell types for modeling (defined in cell_types_for_modeling.pkl)
cell_type_for_modeling_file = f'exp_{EXP_NUM}_cell_types_for_modeling.pkl'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load best hyperparameters from ON/OFF model search (reuse same hyperparams)
hyperparam_path = f'/home/ridvan/Documents/center_surround/outputs/exp_{EXP_NUM}/klindtDedicatedONOFFMixed/exp_{EXP_NUM}_hyperparam_results_latest.pkl'
with open(hyperparam_path, 'rb') as f:
    results = pickle.load(f)

best_params = results['best_params']
print(f"Loaded hyperparameters from: {hyperparam_path}")
print("Best hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

cell_types_path = f'/home/ridvan/Documents/center_surround/data/{cell_type_for_modeling_file}'
with open(cell_types_path, 'rb') as f:
    cell_types_data = pickle.load(f)

all_cells = cell_types_data['all_cells']
cell_type_dict = cell_types_data['cell_types']

# Select cell type to model
# selected_cell_ids = cell_type_dict[cell_type]
# selected_cell_ids = [] 
# selected_cell_ids = [16, 30, 31, 39, 54, 60, 112, 159, 173, 196, 205, 212, 229, 276,
#                         292, 331, 351, 370, 385, 405, 426, 428, 449, 559, 868,
#                         960, 997, 1103, 1118, 1168, 1293, 1356, 1368, 1370, 1418,
#                         1460, 1463, 1477, 1497, 1499, 1535, 1543]
# If you want to model specific cells, you can directly specify their IDs here instead of using the cell type dict. Just make sure they are in the same format as the IDs in all_cells (e.g. "cell_123").
# selected_cell_ids = [148,212,441,478,582,1054,1091,1122,1437,1441,1587]

selected_cell_ids = [ 0, 9, 23, 27, 30, 34, 46, 63, 65, 68, 74, 75, 81, 87, 88, 89, 90,
                    115, 122, 129, 137, 143, 144, 147, 148, 153, 157, 169, 177, 187,
                    202, 206, 211, 212, 213, 214, 221, 232, 235, 256, 258, 269, 276,
                    277, 280, 282, 287, 289, 293, 297, 299, 304, 310, 312, 323, 330,
                    332, 342, 346, 348, 351, 357, 359, 364, 370, 398, 401, 402, 441,
                    460, 466, 468, 473, 478, 480, 485, 488, 489, 491, 493, 497, 500,
                    507, 508, 513, 528, 534, 535, 541, 542, 550, 551, 574, 578, 580,
                    582, 602, 614, 626, 627, 630, 654, 676, 684, 691, 695, 701, 721,
                    723, 729, 745, 755, 778, 789, 790, 807, 825, 838, 864, 869, 882,
                    884, 895, 900, 903, 905, 928, 984, 990, 1004, 1011, 1036, 1046,
                    1053, 1054, 1055, 1059, 1070, 1073, 1075, 1083, 1085, 1087, 1091,
                    1096, 1102, 1109, 1122, 1124, 1138, 1156, 1217, 1231, 1236, 1247,
                    1260, 1262, 1278, 1302, 1311, 1317, 1324, 1357, 1369, 1388, 1389,
                    1392, 1393, 1404, 1408, 1409, 1413, 1422, 1425, 1433, 1437, 1441,
                    1443, 1447, 1451, 1455, 1467, 1487, 1493, 1499, 1506, 1519, 1529,
                    1535, 1538, 1541, 1554, 1559, 1566, 1583, 1587, 1593, 1598, 1604,
                    1613, 1614, 1626, 1640, 1657, 1661, 1662, 1669, 1689, 1690, 1695,
                    1700, 1705, 1707, 1714, 1722, 1730, 1731, 1734, 1735, 1736, 1742,
                    1743, 1746]

if selected_cell_ids is not None:
    neuron_indices = [all_cells.index(cid) for cid in selected_cell_ids]
    picked_cells = selected_cell_ids
else:
    neuron_indices = None
    picked_cells = all_cells

# Load data
data_path = f"/home/ridvan/Documents/center_surround/data/{data_path_file}.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32, normalize_images=True,
                                 neuron_indices=neuron_indices)

# Create output directory with timestamp at the start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'/home/ridvan/Documents/center_surround/outputs/exp_{EXP_NUM}/{model_name}/{cell_type}/run_train_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"\nInput: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Model configuration - using tuned hyperparameters from search
# Using 6 DEDICATED kernels: 2 ON + 2 OFF + 2 Mixed (no kernel sharing between pathways)
model_config = {
    'image_size': input_size,
    'image_channels': in_channels,
    'kernel_sizes': [24, 24],
    'n_on_kernels': 1,     # 2 ON kernels (positive weights) -> dedicated to ON mask
    'n_off_kernels': 1,    # 2 OFF kernels (negative weights) -> dedicated to OFF mask
    'n_mixed_kernels': 0,  # 2 Mixed kernels (1 ON-like, 1 OFF-like) -> dedicated to Mixed mask
    'act_fns': ['relu'],
    'init_scales': [[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    'init_kernels': f"gaussian:0.28",  # single Gaussian (center only)
    'num_neurons': num_neurons,
    'smoothness_reg': best_params['smoothness_reg'],
    'sparsity_reg': 0,
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

# Create model with dedicated kernels per pathway
print("\nCreating model with 6 dedicated kernels:")
print(f"  - Kernels 0-1: ON kernels (positive weights) -> ON mask")
print(f"  - Kernels 2-3: OFF kernels (negative weights) -> OFF mask")
print(f"  - Kernels 4-5: Mixed kernels (1 ON + 1 OFF) -> Mixed mask")

model = KlindtCoreReadoutDedicatedONOFFMixed2D(
    image_size=model_config['image_size'],
    image_channels=model_config['image_channels'],
    kernel_sizes=model_config['kernel_sizes'],
    n_on_kernels=model_config['n_on_kernels'],
    n_off_kernels=model_config['n_off_kernels'],
    n_mixed_kernels=model_config['n_mixed_kernels'],
    act_fns=model_config['act_fns'],
    init_scales=model_config['init_scales'],
    init_kernels=model_config['init_kernels'],
    num_neurons=model_config['num_neurons'],
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
model = train(model, dataloaders, num_epochs=100, lr=best_params['learning_rate'], device=device, early_stopping=True)

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
# print("\nPlotting grid predictions...")
# grid_preds_dir = f'{output_dir}/grid_predictions'
# os.makedirs(grid_preds_dir, exist_ok=True)
# grid_figs = plot_grid_predictions(predictions, targets, correlations)
# for i, fig in enumerate(grid_figs):
#     fig.savefig(f'{grid_preds_dir}/grid_predictions_{i}.png', dpi=150, bbox_inches='tight')
#     plt.close(fig)

# Plot correlation vs reliability (with reliability-dependent plots)
if 'responses_test_by_trial' in raw_data:
    responses_by_trial = raw_data['responses_test_by_trial']
    print(f"Trial data shape (original): {responses_by_trial.shape}")

    # Find neuron axis: matches total number of neurons in the data file
    num_neurons_total = len(all_cells)
    neuron_ax = [ax for ax in range(responses_by_trial.ndim)
                 if responses_by_trial.shape[ax] == num_neurons_total]
    if len(neuron_ax) == 1:
        neuron_ax = neuron_ax[0]
    else:
        raise ValueError(f"Cannot identify neuron axis in responses_test_by_trial "
                         f"(shape {responses_by_trial.shape}, expected one axis with size {num_neurons_total})")

    # Slice selected neurons
    if neuron_indices is not None:
        responses_by_trial = np.take(responses_by_trial, neuron_indices, axis=neuron_ax)

    print(f"Trial data shape (after selection): {responses_by_trial.shape}")

    # bootstrap_reliability expects shape (time, neurons, repetitions)
    # Rearrange so that neuron axis is axis 1
    if neuron_ax == 1:
        pass  # already (time, neurons, reps)
    elif neuron_ax == 0:
        responses_by_trial = responses_by_trial.transpose(1, 0, 2)
    elif neuron_ax == 2:
        responses_by_trial = responses_by_trial.transpose(0, 2, 1)

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
    # foc = correlations / reliability
    # print("\nPlotting fraction of ceiling...")
    # fig_foc = plot_fraction_of_ceiling(foc)
    # fig_foc.savefig(f'{output_dir}/fraction_of_ceiling.png', dpi=150, bbox_inches='tight')
    # plt.close(fig_foc)

    # # Update example predictions with reliability and FoC
    # print("\nPlotting example predictions with reliability...")
    # fig_example_rel = plot_example_predictions(predictions, targets, correlations, reliability, foc)
    # fig_example_rel.savefig(f'{output_dir}/example_predictions_with_reliability.png', dpi=150, bbox_inches='tight')
    # plt.close(fig_example_rel)

# Plot kernels grouped by pathway (ON, OFF, Mixed)
print("\nPlotting kernels by pathway...")
fig3 = plot_kernels_dedicated(model)
fig3.savefig(f'{output_dir}/kernels_by_pathway.png', dpi=150, bbox_inches='tight')

# ============================================================
# Spatial Mask Analysis (from analyze_spatial_masks.py)
# ============================================================
from scripts.klindtDedicatedONOFFMixed.analyze_spatial_masks import (
    plot_spatial_masks_onoff_mixed_separate,
    plot_spatial_masks_combined,
    plot_spatial_masks_with_weights,
    _pathway_polarity,
)

spatial_masks_dir = f'{output_dir}/spatial_masks'
os.makedirs(spatial_masks_dir, exist_ok=True)

# Extract masks from model
mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, 3, num_neurons]
mask_size = model.readout.mask_size
on_masks = mask_weights[:, 0, :].reshape(mask_size[0], mask_size[1], num_neurons)
off_masks = mask_weights[:, 1, :].reshape(mask_size[0], mask_size[1], num_neurons)
mixed_masks = mask_weights[:, 2, :].reshape(mask_size[0], mask_size[1], num_neurons)

# 3) Polarity-weighted masks (uses kernel center values × readout weights)
print("Plotting spatial masks by kernel polarity...")
kernels = model.core.conv_layers[0].weight.detach().cpu().numpy()
kH, kW = kernels.shape[2], kernels.shape[3]
center_h, center_w = kH // 2, kW // 2

n_on = model.core.n_on_kernels
n_off = model.core.n_off_kernels
n_mixed = model.core.n_mixed_kernels

on_kernel_centers = kernels[:n_on, 0, center_h, center_w]
off_kernel_centers = kernels[n_on:n_on+n_off, 0, center_h, center_w]
mixed_kernel_centers = kernels[n_on+n_off:, 0, center_h, center_w]

on_weights_readout = model.readout.on_weights.detach().cpu().numpy()
off_weights_readout = model.readout.off_weights.detach().cpu().numpy()
mixed_weights_readout = model.readout.mixed_weights.detach().cpu().numpy()

figs_pol = plot_spatial_masks_with_weights(
    on_masks, off_masks, mixed_masks,
    on_weights_readout, off_weights_readout, mixed_weights_readout,
    on_kernel_centers, off_kernel_centers, mixed_kernel_centers,
    n_on, n_off, n_mixed,
    neurons_per_plot=8, output_dir=spatial_masks_dir,
)
for fig in figs_pol:
    plt.close(fig)

print(f"All spatial mask plots saved to {spatial_masks_dir}/")

# ============================================================
# Spatial Mask Statistics — TRUE polarity (saved to txt)
# Uses center_value × readout_weight to determine actual ON/OFF,
# not just which mask index the pathway was initialized as.
# ============================================================
noise_threshold = 0.1
polarity_threshold = 0.005

def _pol_label(has_on, has_off):
    if has_on and has_off: return "ON+OFF"
    elif has_on: return "ON"
    elif has_off: return "OFF"
    return "none"

active_pathways_header = f"Active pathways: ON={n_on} OFF={n_off} Mixed={n_mixed}"
stats_lines = []
stats_lines.append("=" * 60)
stats_lines.append("PER-NEURON STATISTICS (true polarity from center × weight)")
stats_lines.append(active_pathways_header)
stats_lines.append("=" * 60)

on_count = 0
off_count = 0
onoff_count = 0

for ni in range(num_neurons):
    # Only consider masks for active pathways
    mask_maxes = {}
    is_signal = {}

    if n_on > 0:
        mask_maxes['ON'] = on_masks[:, :, ni].max()
    if n_off > 0:
        mask_maxes['OFF'] = off_masks[:, :, ni].max()
    if n_mixed > 0:
        mask_maxes['Mixed'] = mixed_masks[:, :, ni].max()

    strongest_max = max(mask_maxes.values()) if mask_maxes else 0
    for key, val in mask_maxes.items():
        is_signal[key] = val >= noise_threshold * strongest_max if strongest_max > 0 else False

    # Compute true polarity of each active pathway using center × weight
    total_on = 0.0
    total_off = 0.0
    path_parts = []

    if n_on > 0:
        on_on, on_off, on_has_on, on_has_off = _pathway_polarity(
            on_kernel_centers, on_weights_readout[:, ni], polarity_threshold)
        on_max = mask_maxes['ON']
        on_sig = is_signal.get('ON', False)
        if on_sig:
            total_on += on_on * on_max
            total_off += on_off * on_max
        path_parts.append(f"ON-path({_pol_label(on_has_on, on_has_off):6s}) mask={on_max:.4f}{'*' if on_sig else ' '}")

    if n_off > 0:
        off_on, off_off, off_has_on, off_has_off = _pathway_polarity(
            off_kernel_centers, off_weights_readout[:, ni], polarity_threshold)
        off_max = mask_maxes['OFF']
        off_sig = is_signal.get('OFF', False)
        if off_sig:
            total_on += off_on * off_max
            total_off += off_off * off_max
        path_parts.append(f"OFF-path({_pol_label(off_has_on, off_has_off):6s}) mask={off_max:.4f}{'*' if off_sig else ' '}")

    if n_mixed > 0:
        mix_on, mix_off, mix_has_on, mix_has_off = _pathway_polarity(
            mixed_kernel_centers, mixed_weights_readout[:, ni], polarity_threshold)
        mixed_max = mask_maxes['Mixed']
        mix_sig = is_signal.get('Mixed', False)
        if mix_sig:
            total_on += mix_on * mixed_max
            total_off += mix_off * mixed_max
        path_parts.append(f"Mix-path({_pol_label(mix_has_on, mix_has_off):6s}) mask={mixed_max:.4f}{'*' if mix_sig else ' '}")

    # Classify neuron by true effective polarity
    if total_on == 0 and total_off == 0:
        true_type = "None"
    elif total_off < 0.1 * total_on:
        true_type = "ON"
        on_count += 1
    elif total_on < 0.1 * total_off:
        true_type = "OFF"
        off_count += 1
    else:
        true_type = "ON-OFF"
        onoff_count += 1

    line = f"Neuron {ni:2d}: Type={true_type:5s} | " + " | ".join(path_parts)
    line += f" | total ON={total_on:.4f} OFF={total_off:.4f}"
    stats_lines.append(line)

stats_lines.append("")
stats_lines.append("=" * 60)
stats_lines.append("SUMMARY (by true polarity)")
stats_lines.append("=" * 60)
stats_lines.append(f"Total neurons: {num_neurons}")
stats_lines.append(f"ON neurons:    {on_count} ({100*on_count/num_neurons:.1f}%)")
stats_lines.append(f"OFF neurons:   {off_count} ({100*off_count/num_neurons:.1f}%)")
stats_lines.append(f"ON-OFF neurons: {onoff_count} ({100*onoff_count/num_neurons:.1f}%)")
none_count = num_neurons - on_count - off_count - onoff_count
if none_count > 0:
    stats_lines.append(f"No signal:     {none_count} ({100*none_count/num_neurons:.1f}%)")

stats_text = "\n".join(stats_lines)
print(stats_text)

with open(f'{spatial_masks_dir}/spatial_mask_stats.txt', 'w') as f:
    f.write(stats_text + "\n")
print(f"Saved: {spatial_masks_dir}/spatial_mask_stats.txt")

# Compute LSTA on NATURAL IMAGES (not test images)
print("\nComputing LSTA...")

def center_crop(image, crop_size=432):
    """Center crop an image to crop_size x crop_size."""
    h, w = image.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top:top+crop_size, left:left+crop_size]

img_path = f"/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/stims/images"
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
if EXP_NUM == 12:
    rf_fits_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis/Checkerboard_Analysis_rec_1/sta_data_3D_fitted.pkl'
elif EXP_NUM == 13:
    rf_fits_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis/Chirp_Checkerboard_Analysis_mix/sta_data_mix.pkl'

with open(rf_fits_path, 'rb') as f:
    rf_fits = pickle.load(f)

# Load experimental LSTA data
lsta_exp_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis/768_pert/lSTAs_data_total_lstas.pkl'
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

# ============================================================
# Cell ID Cards (one summary PNG per neuron)
# ============================================================
print("\nLoading multi-mask LSTA data for cell ID cards...")

# Load multi-mask images (original + 5 masked conditions)
images_mask_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis/natural_images_alsoWith_masks.pkl'
has_multi_mask = os.path.exists(images_mask_path)

if has_multi_mask:
    with open(images_mask_path, 'rb') as f:
        all_mask_images = pickle.load(f)  # [4, 6, 432, 432]

    num_images_lsta = all_mask_images.shape[0]
    num_masks = all_mask_images.shape[1]
    print(f"Loaded multi-mask images: {all_mask_images.shape} ({num_masks} conditions)")

    # Preprocess images per mask condition
    images_all = {}
    for midx in range(num_masks):
        imgs = all_mask_images[:, midx, :, :]
        imgs = (imgs - imgs.mean(axis=(1, 2), keepdims=True)) * 5
        imgs_resized = np.zeros((imgs.shape[0], input_size, input_size), dtype=imgs.dtype)
        for idx in range(imgs.shape[0]):
            imgs_resized[idx] = cv2.resize(imgs[idx], (input_size, input_size),
                                           interpolation=cv2.INTER_AREA)
        images_all[midx] = imgs_resized[:, np.newaxis, :, :]

    # Load experimental LSTA for all mask conditions (upsampled to 432x432)
    lsta_exp_all = {}
    for midx in range(num_masks):
        lsta_exp_mask = []
        for cid in picked_cells:
            cell_lstas = all_lsta_exp[midx][cid]  # (4, 72, 72)
            upsampled = np.repeat(np.repeat(cell_lstas, 6, axis=1), 6, axis=2)
            lsta_exp_mask.append(upsampled)
        lsta_exp_all[midx] = np.array(lsta_exp_mask)

    # Load cell batches for masking
    cell_batches_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/stims/spacing_1.3_vicinity_0.2/cell_batches_info.pkl'
    with open(cell_batches_path, 'rb') as f:
        cell_batches = pickle.load(f)

    masks_batches = cell_batches[(1.3, 0.2)]
    masked_cells = [[cid for cid in batch['selected']] for batch in masks_batches]
    print(f"Loaded {len(masked_cells)} mask batches")

    # Load RF fits for masking (online analysis)
    rf_fits_masking_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis_online/Checkerboard_Analysis_rec_0/sta_data_3D_fitted.pkl'
    with open(rf_fits_masking_path, 'rb') as f:
        rf_fits_masking = pickle.load(f)

    all_RFs = [(key, item['center_analyse']['EllipseCoor'])
               for key, item in rf_fits_masking.items()]

    # Load accepted LSTA lookup
    accepted_lstas_path = f'/home/ridvan/Documents/exp/{EXP_NUM}_exp/{EXP_DATE}/Analysis/lSTAs/accepted_lstas_lookup.pkl'
    with open(accepted_lstas_path, 'rb') as f:
        accepted_lstas = pickle.load(f)

    # Create RF masks for all conditions and compute model LSTA
    rf_masks_by_condition = create_batch_masks_for_conditions(
        masked_cells=masked_cells, all_RFs=all_RFs,
        num_neurons=len(picked_cells),
        mask_size=(input_size, input_size), rf_dim=72, ellipse_scale=0.8,
    )

    lsta_model_all = {}
    for midx in range(num_masks):
        lsta_model_all[midx] = compute_lsta_masked(
            model, images_all[midx], rf_masks_by_condition[midx], device=device)
        print(f"  Mask {midx}: model LSTA shape = {lsta_model_all[midx].shape}")

    mask_labels = ['Original', 'Mask 1', 'Mask 2', 'Mask 3', 'Mask 4', 'Mask 5']

    print("\nGenerating cell ID cards...")
    plot_cell_id_cards(
        kernels=kernels[:, 0], n_on=n_on, n_off=n_off, n_mixed=n_mixed,
        on_masks=on_masks, off_masks=off_masks, mixed_masks=mixed_masks,
        on_weights=on_weights_readout, off_weights=off_weights_readout,
        mixed_weights=mixed_weights_readout,
        on_kernel_centers=on_kernel_centers, off_kernel_centers=off_kernel_centers,
        mixed_kernel_centers=mixed_kernel_centers,
        images_all=images_all, lsta_model_all=lsta_model_all,
        lsta_exp_all=lsta_exp_all, mask_labels=mask_labels,
        masked_cells=masked_cells, all_RFs=all_RFs,
        accepted_lstas=accepted_lstas,
        predictions=predictions, targets=targets,
        correlations=correlations, reliability=reliability,
        cell_ids=picked_cells, cell_type=cell_type,
        rf_fits=rf_fits, rf_dim=72,
        cmap='RdBu_r', vmax_thresh=0.6, expon_treat=3, zoom=2.5,
        target_size=432, smooth_sigma=5, hide_non_accepted=True,
        output_dir=output_dir,
    )
else:
    print(f"Multi-mask images not found at {images_mask_path}, skipping cell ID cards")

# Save the trained model
model_path = f'{output_dir}/best_model_dedicated_onoff_mixed.pth'
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
print(f"  - best_model_dedicated_onoff_mixed.pth (trained model weights)")
print(f"  - results.pkl (metrics, predictions, correlations)")
print(f"  - predictions.png (response curves)")
print(f"  - correlation_distribution.png")
print(f"  - example_predictions.png (actual vs predicted scatter)")
print(f"  - grid_predictions_*.png (all neurons actual vs predicted)")
print(f"  - correlation_vs_reliability.png (with trend line)")
print(f"  - reliability_distribution.png")
print(f"  - fraction_of_ceiling.png")
print(f"  - example_predictions_with_reliability.png")
print(f"  - kernels_by_pathway.png (ON, OFF, Mixed kernels grouped)")
print(f"  - spatial_masks_onoff_mixed_*.png (ON mask, OFF mask, Mixed mask per neuron)")
print(f"  - lsta_per_cell/ (LSTA comparison plots)")
print(f"  - cell_id_cards/ (per-cell summary cards)")
