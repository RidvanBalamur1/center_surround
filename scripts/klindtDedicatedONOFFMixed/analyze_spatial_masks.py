import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from center_surround.models.klindtSurround import KlindtCoreReadoutDedicatedONOFFMixed2D
from center_surround.utils.visualization import _pathway_polarity, _pathway_cmap_and_color

if __name__ == "__main__":
    # Path to the trained model run - UPDATE THIS TO YOUR RUN
    run_dir = "/home/ridvan/Documents/center_surround/outputs/exp_13_data_4/klindtDedicatedONOFFMixed/run_20260223_110750"

    # Load config to get model parameters
    with open(f'{run_dir}/config.pkl', 'rb') as f:
        config = pickle.load(f)

    # Recreate model and load weights
    model_config = config['model_config']
    model = KlindtCoreReadoutDedicatedONOFFMixed2D(**model_config)
    model.load_state_dict(torch.load(f'{run_dir}/best_model_dedicated_onoff_mixed.pth', map_location='cpu'))
    model.eval()

    # ============================================================
    # Get Spatial Masks
    # ============================================================
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, 3, num_neurons]
    mask_size = model.readout.mask_size
    num_neurons = mask_weights.shape[2]

    print(f"Mask weights shape: {mask_weights.shape}")
    print(f"  - {mask_weights.shape[0]} spatial pixels ({mask_size[0]}x{mask_size[1]})")
    print(f"  - 3 masks: ON (index 0), OFF (index 1), Mixed (index 2)")
    print(f"  - {num_neurons} neurons")

    # Reshape masks for visualization
    on_masks = mask_weights[:, 0, :].reshape(mask_size[0], mask_size[1], num_neurons)    # [H, W, N]
    off_masks = mask_weights[:, 1, :].reshape(mask_size[0], mask_size[1], num_neurons)   # [H, W, N]
    mixed_masks = mask_weights[:, 2, :].reshape(mask_size[0], mask_size[1], num_neurons) # [H, W, N]

    print(f"\nON masks shape: {on_masks.shape}")
    print(f"OFF masks shape: {off_masks.shape}")
    print(f"Mixed masks shape: {mixed_masks.shape}")

# ============================================================
# Plot Spatial Masks with Shared Scale (3 columns)
# ============================================================
def plot_spatial_masks_onoff_mixed_separate(on_masks, off_masks, mixed_masks, neurons_per_plot=8, output_dir=None):
    """
    Plot ON/OFF/Mixed spatial masks with shared scaling per neuron.

    Args:
        on_masks: [H, W, N] array of ON masks
        off_masks: [H, W, N] array of OFF masks
        mixed_masks: [H, W, N] array of Mixed masks
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

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        fig, axes = plt.subplots(n_neurons, 3, figsize=(10, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            on_mask = on_masks[:, :, neuron_idx]
            off_mask = off_masks[:, :, neuron_idx]
            mixed_mask = mixed_masks[:, :, neuron_idx]

            # Shared scale for this neuron
            vmax = max(on_mask.max(), off_mask.max(), mixed_mask.max())
            if vmax == 0:
                vmax = 1

            # Column 0: ON mask
            ax = axes[i, 0]
            im = ax.imshow(on_mask, cmap='Reds', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('ON Mask', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Column 1: OFF mask
            ax = axes[i, 1]
            im = ax.imshow(off_mask, cmap='Blues', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('OFF Mask', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Column 2: Mixed mask
            ax = axes[i, 2]
            im = ax.imshow(mixed_mask, cmap='Greens', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('Mixed Mask', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'ON/OFF/Mixed Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

        if output_dir:
            fig.savefig(f'{output_dir}/spatial_masks_onoff_mixed_{batch_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/spatial_masks_onoff_mixed_{batch_idx}.png")

    return figures

# ============================================================
# Plot Combined ON/OFF/Mixed Masks (RGB Overlay)
# ============================================================
def plot_spatial_masks_combined(on_masks, off_masks, mixed_masks, neurons_per_plot=8, output_dir=None, noise_ratio_threshold=0.1):
    """
    Plot ON, OFF, and Mixed masks overlaid on top of each other using different colors.

    ON mask: Red channel
    OFF mask: Blue channel
    Mixed mask: Green channel
    Overlaps show color combinations (e.g., Red+Blue=Magenta, Red+Green=Yellow, etc.)

    Args:
        on_masks: [H, W, N] array of ON masks
        off_masks: [H, W, N] array of OFF masks
        mixed_masks: [H, W, N] array of Mixed masks
        neurons_per_plot: Number of neurons per figure
        output_dir: Directory to save figures (if None, displays instead)
        noise_ratio_threshold: Per-neuron threshold. A mask is considered noise if its max value
                               is less than this fraction of the strongest mask's max value.

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

        # 5 columns: ON (gray), OFF (gray), Mixed (gray), Combined (RGB), Combined (contour)
        fig, axes = plt.subplots(n_neurons, 5, figsize=(18, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            on_mask = on_masks[:, :, neuron_idx]
            off_mask = off_masks[:, :, neuron_idx]
            mixed_mask = mixed_masks[:, :, neuron_idx]

            on_max = on_mask.max()
            off_max = off_mask.max()
            mixed_max = mixed_mask.max()

            # Determine if each mask is signal or noise (per-neuron comparison)
            strongest_max = max(on_max, off_max, mixed_max)
            if strongest_max == 0:
                on_is_signal = False
                off_is_signal = False
                mixed_is_signal = False
            else:
                on_is_signal = on_max >= noise_ratio_threshold * strongest_max
                off_is_signal = off_max >= noise_ratio_threshold * strongest_max
                mixed_is_signal = mixed_max >= noise_ratio_threshold * strongest_max

            # Shared scale for grayscale plots
            vmax = strongest_max
            if vmax == 0:
                vmax = 1

            # Column 0: ON mask (grayscale)
            ax = axes[i, 0]
            im = ax.imshow(on_mask, cmap='gray', vmin=0, vmax=vmax)
            title_suffix = "" if on_is_signal else " (noise)"
            if i == 0:
                ax.set_title('ON Mask', fontsize=12, fontweight='bold')
            signal_count = sum([on_is_signal, off_is_signal, mixed_is_signal])
            ax.set_ylabel(f'Neuron {neuron_idx}\n({signal_count}/3 active)', fontsize=10,
                         color='black' if signal_count > 0 else 'gray')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 1: OFF mask (grayscale)
            ax = axes[i, 1]
            im = ax.imshow(off_mask, cmap='gray', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('OFF Mask', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 2: Mixed mask (grayscale)
            ax = axes[i, 2]
            im = ax.imshow(mixed_mask, cmap='gray', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('Mixed Mask', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 3: Combined RGB (ON=Red, OFF=Blue, Mixed=Green)
            ax = axes[i, 3]
            rgb = np.zeros((H, W, 3))
            if on_is_signal:
                on_norm = on_mask / (on_max + 1e-8)
                rgb[:, :, 0] = on_norm   # Red = ON
            if mixed_is_signal:
                mixed_norm = mixed_mask / (mixed_max + 1e-8)
                rgb[:, :, 1] = mixed_norm  # Green = Mixed
            if off_is_signal:
                off_norm = off_mask / (off_max + 1e-8)
                rgb[:, :, 2] = off_norm  # Blue = OFF
            ax.imshow(rgb)
            if i == 0:
                ax.set_title('Combined (R=ON, G=Mixed, B=OFF)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 4: Contour overlay
            ax = axes[i, 4]
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

            # Plot Mixed contours in green (only if signal)
            if mixed_is_signal:
                levels_mixed = [mixed_max * t for t in [0.2, 0.5, 0.8]]
                ax.contour(mixed_mask, levels=levels_mixed, colors='green', linewidths=1.5)
                ax.contourf(mixed_mask, levels=[levels_mixed[1], mixed_max], colors=['green'], alpha=0.3)

            # Add text if no masks have signal
            if not on_is_signal and not off_is_signal and not mixed_is_signal:
                ax.text(W/2, H/2, 'No signal', ha='center', va='center',
                       fontsize=10, color='gray', style='italic')

            if i == 0:
                ax.set_title('Contours (R=ON, G=Mixed, B=OFF)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f'ON/OFF/Mixed Spatial Segregation (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

        if output_dir:
            fig.savefig(f'{output_dir}/spatial_masks_combined_{batch_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/spatial_masks_combined_{batch_idx}.png")

    return figures


# ============================================================
# Plot Spatial Masks Color-Coded by Kernel Polarity
# ============================================================
def _pathway_label(has_on, has_off, on_part, off_part):
    """Return a net polarity label string."""
    if has_on and has_off:
        return f"ON+OFF ({on_part:.4f}, -{off_part:.4f})"
    elif has_on:
        return f"ON ({on_part:.4f})"
    elif has_off:
        return f"OFF (-{off_part:.4f})"
    return "none"


def plot_spatial_masks_with_weights(on_masks, off_masks, mixed_masks,
                                    on_weights, off_weights, mixed_weights,
                                    on_kernel_centers, off_kernel_centers, mixed_kernel_centers,
                                    n_on, n_off, n_mixed,
                                    neurons_per_plot=8, output_dir=None,
                                    noise_ratio_threshold=0.1,
                                    polarity_threshold=0.005):
    """
    Plot spatial masks color-coded by the effective polarity of their kernels.

    Per-kernel contribution: center_value[k] * readout_weight[k].
    Positive products are ON contributions, negative are OFF contributions.
    Each pathway can have pure ON (red), pure OFF (blue), or mixed ON+OFF (green).

    The combined polarity map splits contributions per-kernel across all pathways:
    - Red = pure ON spatial activity
    - Blue = pure OFF spatial activity
    - Green = ON and OFF overlap at the same spatial location

    Columns:
    - ON/OFF/Mixed pathway masks: Reds if pure ON, Blues if pure OFF, Greens if mixed
    - Combined polarity map: per-kernel split (Red=ON, Green=overlap, Blue=OFF)
    - Contour overlay: pathway contours colored by polarity
    """
    num_neurons = on_masks.shape[2]
    H, W = on_masks.shape[:2]

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        fig, axes = plt.subplots(n_neurons, 5, figsize=(20, 3.5*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            on_mask = on_masks[:, :, neuron_idx]
            off_mask = off_masks[:, :, neuron_idx]
            mixed_mask = mixed_masks[:, :, neuron_idx]

            on_max = on_mask.max()
            off_max = off_mask.max()
            mixed_max = mixed_mask.max()

            # Determine signal vs noise per mask
            strongest_max = max(on_max, off_max, mixed_max)
            if strongest_max == 0:
                on_is_signal = off_is_signal = mixed_is_signal = False
            else:
                on_is_signal = on_max >= noise_ratio_threshold * strongest_max
                off_is_signal = off_max >= noise_ratio_threshold * strongest_max
                mixed_is_signal = mixed_max >= noise_ratio_threshold * strongest_max

            vmax = strongest_max if strongest_max > 0 else 1

            # Compute per-kernel polarity for each pathway
            on_on, on_off, on_has_on, on_has_off = _pathway_polarity(
                on_kernel_centers, on_weights[:, neuron_idx], polarity_threshold)
            off_on, off_off, off_has_on, off_has_off = _pathway_polarity(
                off_kernel_centers, off_weights[:, neuron_idx], polarity_threshold)
            mix_on, mix_off, mix_has_on, mix_has_off = _pathway_polarity(
                mixed_kernel_centers, mixed_weights[:, neuron_idx], polarity_threshold)

            # Build weight annotation strings
            on_parts = [f"K{k}({'ON' if on_kernel_centers[k] >= 0 else 'OFF'}): {on_weights[k, neuron_idx]:.3f}"
                        for k in range(n_on)]
            on_w_str = " | ".join(on_parts) + f"\nnet: {_pathway_label(on_has_on, on_has_off, on_on, on_off)}"

            off_parts = [f"K{k}({'ON' if off_kernel_centers[k] >= 0 else 'OFF'}): {off_weights[k, neuron_idx]:.3f}"
                         for k in range(n_off)]
            off_w_str = " | ".join(off_parts) + f"\nnet: {_pathway_label(off_has_on, off_has_off, off_on, off_off)}"

            mixed_parts = [f"K{k}({'ON' if mixed_kernel_centers[k] >= 0 else 'OFF'}): {mixed_weights[k, neuron_idx]:.3f}"
                           for k in range(n_mixed)]
            mixed_w_str = " | ".join(mixed_parts) + f"\nnet: {_pathway_label(mix_has_on, mix_has_off, mix_on, mix_off)}"

            # Column 0: ON pathway mask
            ax = axes[i, 0]
            cmap, txtcolor = _pathway_cmap_and_color(on_has_on, on_has_off)
            ax.imshow(on_mask, cmap=cmap, vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('ON Pathway', fontsize=12, fontweight='bold')
            signal_count = sum([on_is_signal, off_is_signal, mixed_is_signal])
            ax.set_ylabel(f'Neuron {neuron_idx}\n({signal_count}/3 active)', fontsize=10,
                         color='black' if signal_count > 0 else 'gray')
            ax.set_xlabel(on_w_str, fontsize=7, color=txtcolor)
            ax.set_xticks([]); ax.set_yticks([])

            # Column 1: OFF pathway mask
            ax = axes[i, 1]
            cmap, txtcolor = _pathway_cmap_and_color(off_has_on, off_has_off)
            ax.imshow(off_mask, cmap=cmap, vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('OFF Pathway', fontsize=12, fontweight='bold')
            ax.set_xlabel(off_w_str, fontsize=7, color=txtcolor)
            ax.set_xticks([]); ax.set_yticks([])

            # Column 2: Mixed pathway mask
            ax = axes[i, 2]
            cmap, txtcolor = _pathway_cmap_and_color(mix_has_on, mix_has_off)
            ax.imshow(mixed_mask, cmap=cmap, vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title('Mixed Pathway', fontsize=12, fontweight='bold')
            ax.set_xlabel(mixed_w_str, fontsize=7, color=txtcolor)
            ax.set_xticks([]); ax.set_yticks([])

            # Column 3: Combined polarity map (per-kernel split)
            # Red = pure ON, Green = ON+OFF overlap, Blue = pure OFF
            ax = axes[i, 3]
            on_channel = np.zeros((H, W))
            off_channel = np.zeros((H, W))

            # Split per-kernel across all pathways
            for pathway_mask, centers, weights_n, is_signal in [
                (on_mask, on_kernel_centers, on_weights[:, neuron_idx], on_is_signal),
                (off_mask, off_kernel_centers, off_weights[:, neuron_idx], off_is_signal),
                (mixed_mask, mixed_kernel_centers, mixed_weights[:, neuron_idx], mixed_is_signal),
            ]:
                if not is_signal:
                    continue
                for k in range(len(centers)):
                    if abs(weights_n[k]) < polarity_threshold:
                        continue  # kernel not used by this neuron
                    contrib = centers[k] * weights_n[k]
                    if contrib >= 0:
                        on_channel += pathway_mask * contrib
                    else:
                        off_channel += pathway_mask * abs(contrib)

            max_val = max(on_channel.max(), off_channel.max(), 1e-8)
            on_norm = on_channel / max_val
            off_norm = off_channel / max_val

            # Per-pixel classification: each pixel gets exactly one color
            # Red = pure ON, Green = ON+OFF overlap, Blue = pure OFF
            # Threshold each channel relative to its own peak so the weaker
            # channel isn't drowned out by the stronger one
            on_peak = on_channel.max()
            off_peak = off_channel.max()
            on_present = on_channel > 0.05 * on_peak if on_peak > 0 else np.full((H, W), False)
            off_present = off_channel > 0.05 * off_peak if off_peak > 0 else np.full((H, W), False)
            both = on_present & off_present
            on_only_px = on_present & ~off_present
            off_only_px = off_present & ~on_present

            rgb = np.zeros((H, W, 3))
            rgb[on_only_px, 0] = on_norm[on_only_px]             # Red = pure ON
            rgb[off_only_px, 2] = off_norm[off_only_px]           # Blue = pure OFF
            rgb[both, 1] = np.maximum(on_norm[both], off_norm[both])  # Green = overlap
            ax.imshow(rgb)
            if i == 0:
                ax.set_title('Combined Polarity\n(R=ON, G=ON+OFF, B=OFF)', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

            # Column 4: Contour overlay colored by polarity
            ax = axes[i, 4]
            ax.imshow(np.ones((H, W, 3)) * 0.9)

            for mask, has_on, has_off, is_signal, mask_max in [
                (on_mask, on_has_on, on_has_off, on_is_signal, on_max),
                (off_mask, off_has_on, off_has_off, off_is_signal, off_max),
                (mixed_mask, mix_has_on, mix_has_off, mixed_is_signal, mixed_max),
            ]:
                if not is_signal or mask_max == 0:
                    continue
                if has_on and has_off:
                    color = 'green'
                elif has_on:
                    color = 'red'
                else:
                    color = 'blue'
                levels = [mask_max * t for t in [0.2, 0.5, 0.8]]
                ax.contour(mask, levels=levels, colors=color, linewidths=1.5)
                ax.contourf(mask, levels=[levels[1], mask_max], colors=[color], alpha=0.3)

            if not on_is_signal and not off_is_signal and not mixed_is_signal:
                ax.text(W/2, H/2, 'No signal', ha='center', va='center',
                       fontsize=10, color='gray', style='italic')

            if i == 0:
                ax.set_title('Contours\n(R=ON, G=ON+OFF, B=OFF)', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

        plt.suptitle(f'Spatial Masks by Kernel Polarity (Neurons {neuron_indices[0]}-{neuron_indices[-1]})\n'
                     f'Color from center × weight: Red=ON, Green=ON+OFF, Blue=OFF', fontsize=13)
        plt.tight_layout()
        figures.append(fig)

        if output_dir:
            fig.savefig(f'{output_dir}/spatial_masks_with_weights_{batch_idx}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/spatial_masks_with_weights_{batch_idx}.png")

    return figures


if __name__ == "__main__":
    # Create output directory for new plots
    output_dir = f'{run_dir}/spatial_masks_reanalysis'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving plots to: {output_dir}")

    # Plot separate ON/OFF/Mixed masks
    print("\nPlotting separate ON/OFF/Mixed masks...")
    figs = plot_spatial_masks_onoff_mixed_separate(on_masks, off_masks, mixed_masks, neurons_per_plot=8, output_dir=output_dir)
    print(f"Saved {len(figs)} separate mask figure(s)")

    # Plot combined ON/OFF/Mixed masks
    print("\nPlotting combined ON/OFF/Mixed masks...")
    figs_combined = plot_spatial_masks_combined(on_masks, off_masks, mixed_masks, neurons_per_plot=8, output_dir=output_dir)
    print(f"Saved {len(figs_combined)} combined mask figure(s)")

    # ============================================================
    # Plot Spatial Masks Color-Coded by Kernel Polarity
    # ============================================================
    # Extract kernel center values to determine true ON/OFF polarity
    kernels = model.core.conv_layers[0].weight.detach().cpu().numpy()  # [n_total, C_in, kH, kW]
    kH, kW = kernels.shape[2], kernels.shape[3]
    center_h, center_w = kH // 2, kW // 2

    n_on = model.core.n_on_kernels
    n_off = model.core.n_off_kernels
    n_mixed = model.core.n_mixed_kernels

    on_kernel_centers = kernels[:n_on, 0, center_h, center_w]
    off_kernel_centers = kernels[n_on:n_on+n_off, 0, center_h, center_w]
    mixed_kernel_centers = kernels[n_on+n_off:, 0, center_h, center_w]

    # Extract readout weights for each pathway
    on_weights_readout = model.readout.on_weights.detach().cpu().numpy()  # [n_on, N]
    off_weights_readout = model.readout.off_weights.detach().cpu().numpy()  # [n_off, N]
    mixed_weights_readout = model.readout.mixed_weights.detach().cpu().numpy()  # [n_mixed, N]

    print(f"\nKernel center values (center pixel at [{center_h}, {center_w}]):")
    for k in range(n_on):
        print(f"  ON K{k}: center={on_kernel_centers[k]:.4f} -> {'ON' if on_kernel_centers[k] >= 0 else 'OFF'}")
    for k in range(n_off):
        print(f"  OFF K{k}: center={off_kernel_centers[k]:.4f} -> {'ON' if off_kernel_centers[k] >= 0 else 'OFF'}")
    for k in range(n_mixed):
        print(f"  Mixed K{k}: center={mixed_kernel_centers[k]:.4f} -> {'ON' if mixed_kernel_centers[k] >= 0 else 'OFF'}")

    print("\nPlotting spatial masks by kernel polarity...")
    figs_weights = plot_spatial_masks_with_weights(
        on_masks, off_masks, mixed_masks,
        on_weights_readout, off_weights_readout, mixed_weights_readout,
        on_kernel_centers, off_kernel_centers, mixed_kernel_centers,
        n_on, n_off, n_mixed,
        neurons_per_plot=8, output_dir=output_dir
    )
    print(f"Saved {len(figs_weights)} polarity mask figure(s)")

    # ============================================================
    # Print Statistics
    # ============================================================
    print("\n" + "="*50)
    print("PER-NEURON STATISTICS")
    print("="*50)

    noise_threshold = 0.1

    for neuron_idx in range(num_neurons):
        on_mask = on_masks[:, :, neuron_idx]
        off_mask = off_masks[:, :, neuron_idx]
        mixed_mask = mixed_masks[:, :, neuron_idx]

        on_max = on_mask.max()
        off_max = off_mask.max()
        mixed_max = mixed_mask.max()
        strongest_max = max(on_max, off_max, mixed_max)

        if strongest_max == 0:
            dominant = "None"
            on_is_signal = off_is_signal = mixed_is_signal = False
        else:
            on_is_signal = on_max >= noise_threshold * strongest_max
            off_is_signal = off_max >= noise_threshold * strongest_max
            mixed_is_signal = mixed_max >= noise_threshold * strongest_max

            # Determine dominant pathway
            if on_max >= off_max and on_max >= mixed_max:
                dominant = "ON"
            elif off_max >= on_max and off_max >= mixed_max:
                dominant = "OFF"
            else:
                dominant = "Mixed"

        active_pathways = []
        if on_is_signal:
            active_pathways.append("ON")
        if off_is_signal:
            active_pathways.append("OFF")
        if mixed_is_signal:
            active_pathways.append("Mixed")

        print(f"Neuron {neuron_idx:2d}: Dominant={dominant:5s} | "
              f"ON={on_max:.4f}{'*' if on_is_signal else ' '} | "
              f"OFF={off_max:.4f}{'*' if off_is_signal else ' '} | "
              f"Mixed={mixed_max:.4f}{'*' if mixed_is_signal else ' '} | "
              f"Active: {', '.join(active_pathways) if active_pathways else 'None'}")

    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    on_dominant = 0
    off_dominant = 0
    mixed_dominant = 0
    multi_pathway = 0

    for neuron_idx in range(num_neurons):
        on_mask = on_masks[:, :, neuron_idx]
        off_mask = off_masks[:, :, neuron_idx]
        mixed_mask = mixed_masks[:, :, neuron_idx]

        on_max = on_mask.max()
        off_max = off_mask.max()
        mixed_max = mixed_mask.max()
        strongest_max = max(on_max, off_max, mixed_max)

        if strongest_max > 0:
            on_is_signal = on_max >= noise_threshold * strongest_max
            off_is_signal = off_max >= noise_threshold * strongest_max
            mixed_is_signal = mixed_max >= noise_threshold * strongest_max

            signal_count = sum([on_is_signal, off_is_signal, mixed_is_signal])
            if signal_count > 1:
                multi_pathway += 1

            if on_max >= off_max and on_max >= mixed_max:
                on_dominant += 1
            elif off_max >= on_max and off_max >= mixed_max:
                off_dominant += 1
            else:
                mixed_dominant += 1

    print(f"Total neurons: {num_neurons}")
    print(f"ON-dominant neurons: {on_dominant} ({100*on_dominant/num_neurons:.1f}%)")
    print(f"OFF-dominant neurons: {off_dominant} ({100*off_dominant/num_neurons:.1f}%)")
    print(f"Mixed-dominant neurons: {mixed_dominant} ({100*mixed_dominant/num_neurons:.1f}%)")
    print(f"Multi-pathway neurons (>1 active): {multi_pathway} ({100*multi_pathway/num_neurons:.1f}%)")

    print(f"\nAll plots saved to: {output_dir}")
