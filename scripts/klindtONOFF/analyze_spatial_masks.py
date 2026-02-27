import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from center_surround.models.klindtSurround import KlindtCoreReadoutONOFF2D

# Path to the trained model run
run_dir = "/home/ridvan/Documents/center_surround/outputs/exp_13_data_4/klindtONOFF/run_20260127_152654"

# Load config to get model parameters
with open(f'{run_dir}/config.pkl', 'rb') as f:
    config = pickle.load(f)

# Recreate model and load weights
model_config = config['model_config']
model = KlindtCoreReadoutONOFF2D(**model_config)
model.load_state_dict(torch.load(f'{run_dir}/best_model_onoff.pth', map_location='cpu'))
model.eval()

# ============================================================
# Get Spatial Masks
# ============================================================
mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, 2, num_neurons]
mask_size = model.readout.mask_size
num_neurons = mask_weights.shape[2]

print(f"Mask weights shape: {mask_weights.shape}")
print(f"  - {mask_weights.shape[0]} spatial pixels ({mask_size[0]}x{mask_size[1]})")
print(f"  - 2 masks: ON (index 0), OFF (index 1)")
print(f"  - {num_neurons} neurons")

# Reshape masks for visualization
on_masks = mask_weights[:, 0, :].reshape(mask_size[0], mask_size[1], num_neurons)   # [H, W, N]
off_masks = mask_weights[:, 1, :].reshape(mask_size[0], mask_size[1], num_neurons)  # [H, W, N]

print(f"\nON masks shape: {on_masks.shape}")
print(f"OFF masks shape: {off_masks.shape}")

# ============================================================
# Plot Spatial Masks with Shared Scale
# ============================================================
def plot_spatial_masks_onoff(on_masks, off_masks, neurons_per_plot=8, output_dir=None):
    """
    Plot ON/OFF spatial masks with shared scaling per neuron.

    Args:
        on_masks: [H, W, N] array of ON masks
        off_masks: [H, W, N] array of OFF masks
        neurons_per_plot: Number of neurons per figure
        output_dir: Directory to save figures (if None, displays instead)

    Returns:
        List of figures
    """
    num_neurons = on_masks.shape[2]

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []
    mask_labels = ['ON Mask', 'OFF Mask']

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        fig, axes = plt.subplots(n_neurons, 2, figsize=(6, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            on_mask = on_masks[:, :, neuron_idx]
            off_mask = off_masks[:, :, neuron_idx]

            # Shared scale: vmin=0, vmax=max of both masks
            vmin = 0
            vmax = max(on_mask.max(), off_mask.max())
            if vmax == 0:
                vmax = 1

            masks = [on_mask, off_mask]
            for m in range(2):
                ax = axes[i, m]
                im = ax.imshow(masks[m], cmap='gray', vmin=vmin, vmax=vmax)
                if i == 0:
                    ax.set_title(mask_labels[m], fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'ON/OFF Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

        if output_dir:
            fig.savefig(f'{output_dir}/spatial_masks_onoff_{batch_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/spatial_masks_onoff_{batch_idx}.png")

    return figures

# ============================================================
# Plot Combined ON/OFF Masks (Overlay)
# ============================================================
def plot_spatial_masks_combined(on_masks, off_masks, neurons_per_plot=8, output_dir=None, noise_ratio_threshold=0.1):
    """
    Plot ON and OFF masks overlaid on top of each other using different colors.

    ON mask: Red channel
    OFF mask: Blue channel
    Overlap: Magenta (red + blue)

    Args:
        on_masks: [H, W, N] array of ON masks
        off_masks: [H, W, N] array of OFF masks
        neurons_per_plot: Number of neurons per figure
        output_dir: Directory to save figures (if None, displays instead)
        noise_ratio_threshold: Per-neuron threshold. A mask is considered noise if its max value
                               is less than this fraction of the stronger mask's max value.
                               E.g., 0.1 means if ON max is 10x smaller than OFF max, ON is noise.

    Returns:
        List of figures
    """
    num_neurons = on_masks.shape[2]
    H, W = on_masks.shape[:2]

    print(f"Per-neuron noise detection (ratio threshold: {noise_ratio_threshold})")

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        # 4 columns: ON (gray), OFF (gray), Combined (color), Combined (contour)
        fig, axes = plt.subplots(n_neurons, 4, figsize=(14, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            on_mask = on_masks[:, :, neuron_idx]
            off_mask = off_masks[:, :, neuron_idx]

            on_max = on_mask.max()
            off_max = off_mask.max()

            # Determine if each mask is signal or noise (per-neuron comparison)
            # A mask is considered signal if its max is at least noise_ratio_threshold
            # times the max of the stronger mask
            stronger_max = max(on_max, off_max)
            if stronger_max == 0:
                on_is_signal = False
                off_is_signal = False
            else:
                on_is_signal = on_max >= noise_ratio_threshold * stronger_max
                off_is_signal = off_max >= noise_ratio_threshold * stronger_max

            # Shared scale for grayscale plots
            vmax = max(on_max, off_max)
            if vmax == 0:
                vmax = 1

            # Column 0: ON mask (grayscale)
            ax = axes[i, 0]
            im = ax.imshow(on_mask, cmap='gray', vmin=0, vmax=vmax)
            title_suffix = "" if on_is_signal else " (noise)"
            if i == 0:
                ax.set_title('ON Mask', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Neuron {neuron_idx}{title_suffix}', fontsize=10,
                         color='black' if on_is_signal or off_is_signal else 'gray')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 1: OFF mask (grayscale)
            ax = axes[i, 1]
            im = ax.imshow(off_mask, cmap='gray', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('OFF Mask', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 2: Combined RGB (ON=Red, OFF=Blue)
            # Only show color for masks above noise threshold
            ax = axes[i, 2]
            rgb = np.zeros((H, W, 3))
            if on_is_signal:
                on_norm = on_mask / (on_max + 1e-8)
                rgb[:, :, 0] = on_norm   # Red = ON
            if off_is_signal:
                off_norm = off_mask / (off_max + 1e-8)
                rgb[:, :, 2] = off_norm  # Blue = OFF
            ax.imshow(rgb)
            if i == 0:
                ax.set_title('Combined (R=ON, B=OFF)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 3: Contour overlay
            ax = axes[i, 3]
            ax.imshow(np.ones((H, W, 3)) * 0.9)  # Light gray background

            # Plot ON contours in red (only if signal)
            if on_is_signal:
                levels_on = [on_max * t for t in [0.2, 0.5, 0.8]]
                ax.contour(on_mask, levels=levels_on, colors='red', linewidths=1.5)
                ax.contourf(on_mask, levels=[levels_on[1], on_max], colors=['red'], alpha=0.3)

            # Plot OFF contours in blue (only if signal)
            if off_is_signal:
                levels_off = [off_max * t for t in [0.2, 0.5, 0.8]]
                ax.contour(off_mask, levels=levels_off, colors='blue', linewidths=1.5)
                ax.contourf(off_mask, levels=[levels_off[1], off_max], colors=['blue'], alpha=0.3)

            # Add text if both are noise
            if not on_is_signal and not off_is_signal:
                ax.text(W/2, H/2, 'No signal', ha='center', va='center',
                       fontsize=10, color='gray', style='italic')

            if i == 0:
                ax.set_title('Contours (R=ON, B=OFF)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f'ON/OFF Spatial Segregation (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

        if output_dir:
            fig.savefig(f'{output_dir}/spatial_masks_combined_{batch_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/spatial_masks_combined_{batch_idx}.png")

    return figures


# Create output directory for new plots
output_dir = f'{run_dir}/spatial_masks_reanalysis'
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving plots to: {output_dir}")

# Plot separate ON/OFF masks
print("\nPlotting separate ON/OFF masks...")
figs = plot_spatial_masks_onoff(on_masks, off_masks, neurons_per_plot=8, output_dir=output_dir)
print(f"Saved {len(figs)} separate mask figure(s)")

# Plot combined ON/OFF masks
print("\nPlotting combined ON/OFF masks...")
figs_combined = plot_spatial_masks_combined(on_masks, off_masks, neurons_per_plot=8, output_dir=output_dir)
print(f"Saved {len(figs_combined)} combined mask figure(s)")

# ============================================================
# Print Statistics
# ============================================================
print("\n" + "="*50)
print("PER-NEURON STATISTICS")
print("="*50)

for n in range(min(5, num_neurons)):  # Show first 5 neurons
    on_max = on_masks[:,:,n].max()
    off_max = off_masks[:,:,n].max()
    ratio = on_max / (off_max + 1e-8)
    print(f"Neuron {n}: ON max={on_max:.4f}, OFF max={off_max:.6f}, ratio={ratio:.1f}")

plt.close('all')