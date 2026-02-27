import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_correlation_distribution(correlations: np.ndarray, mean_corr: float = None,
                                   median_corr: float = None):
    """
    Plot histogram of model correlation distribution.

    Args:
        correlations: Array of correlation values per neuron
        mean_corr: Mean correlation (auto-computed if None)
        median_corr: Median correlation (auto-computed if None)

    Returns:
        matplotlib.figure.Figure
    """
    if mean_corr is None:
        mean_corr = np.nanmean(correlations)
    if median_corr is None:
        median_corr = np.nanmedian(correlations)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(mean_corr, color='r', linestyle='--', linewidth=2,
               label=f"Mean: {mean_corr:.4f}")
    ax.axvline(median_corr, color='g', linestyle='--', linewidth=2,
               label=f"Median: {median_corr:.4f}")
    ax.set_title("Model Correlation Distribution")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_reliability_distribution(reliability: np.ndarray):
    """
    Plot histogram of neuron reliability distribution.

    Args:
        reliability: Array of reliability values per neuron

    Returns:
        matplotlib.figure.Figure
    """
    mean_rel = np.nanmean(reliability)
    median_rel = np.nanmedian(reliability)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(reliability, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(mean_rel, color='r', linestyle='--', linewidth=2,
               label=f"Mean: {mean_rel:.4f}")
    ax.axvline(median_rel, color='g', linestyle='--', linewidth=2,
               label=f"Median: {median_rel:.4f}")
    ax.set_title("Neuron Reliability Distribution")
    ax.set_xlabel("Reliability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fraction_of_ceiling(foc: np.ndarray, mean_foc: float = None,
                              median_foc: float = None):
    """
    Plot histogram of fraction of ceiling (correlation / reliability).

    Args:
        foc: Array of FoC values per neuron
        mean_foc: Mean FoC (auto-computed if None)
        median_foc: Median FoC (auto-computed if None)

    Returns:
        matplotlib.figure.Figure
    """
    # Filter out NaN values for plotting
    foc_clean = np.array([f for f in foc if not np.isnan(f)])

    if mean_foc is None:
        mean_foc = np.nanmean(foc)
    if median_foc is None:
        median_foc = np.nanmedian(foc)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(foc_clean, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(mean_foc, color='r', linestyle='--', linewidth=2,
               label=f"Mean: {mean_foc:.4f}")
    ax.axvline(median_foc, color='g', linestyle='--', linewidth=2,
               label=f"Median: {median_foc:.4f}")
    ax.axvline(1.0, color='k', linestyle='--', linewidth=2, label='Ceiling')
    ax.set_title("Fraction of Ceiling Distribution")
    ax.set_xlabel("Model Correlation / Reliability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def bootstrap_reliability(data: np.ndarray, n_bootstrap: int = 100, axis: int = 2):
    """
    Calculate reliability using bootstrap sampling.
    
    Args:
        data: shape (time, neurons, repetitions)
    
    Returns:
        reliability: [num_neurons]
        reliability_ci: [2, num_neurons] - 2.5% and 97.5% confidence intervals
    """
    n_reps = data.shape[axis]
    reliabilities = []
    
    for _ in range(n_bootstrap):
        idx1 = np.random.choice(n_reps, size=n_reps//2, replace=True)
        idx2 = np.random.choice(list(set(range(n_reps)) - set(idx1)), 
                              size=min(n_reps//2, len(set(range(n_reps)) - set(idx1))), 
                              replace=True)
        
        m1 = np.mean(data[:, :, idx1], axis=axis)
        m2 = np.mean(data[:, :, idx2], axis=axis)
        
        corrs = [np.corrcoef(m1[:, n], m2[:, n])[0, 1] for n in range(data.shape[1])]
        reliabilities.append(corrs)
    
    reliability = np.mean(reliabilities, axis=0)
    reliability_ci = np.percentile(reliabilities, [2.5, 97.5], axis=0)
    return reliability, reliability_ci


def plot_predictions_grid(predictions: np.ndarray, targets: np.ndarray,
                          correlations: np.ndarray, neuron_indices: Optional[List[int]] = None):
    """
    Plot grid of predicted vs actual responses for all neurons.

    Args:
        predictions: Model output, shape [num_samples, num_neurons]
        targets: Ground truth values, same shape as predictions
        correlations: Correlation values per neuron
        neuron_indices: Optional list of neuron indices to visualize (default: all)

    Returns:
        matplotlib.figure.Figure
    """
    from math import sqrt, ceil

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    num_samples, total_neurons = predictions.shape

    if neuron_indices is None:
        neuron_indices = list(range(total_neurons))
    else:
        neuron_indices = [i for i in neuron_indices if 0 <= i < total_neurons]

    n_cells = len(neuron_indices)
    grid_cols = min(8, ceil(sqrt(n_cells)))
    grid_rows = ceil(n_cells / grid_cols)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(2.5 * grid_cols, 2 * grid_rows))

    if n_cells == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_cells:
            neuron_idx = neuron_indices[i]
            ax.plot(targets[:, neuron_idx], label='Target', linestyle='--', marker='o', alpha=0.7, markersize=3)
            ax.plot(predictions[:, neuron_idx], label='Output', linestyle='-', marker='x', alpha=0.7, markersize=3)
            ax.set_title(f"Neuron {neuron_idx} (r={correlations[neuron_idx]:.3f})")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Response")
            ax.legend(fontsize=8)
        else:
            ax.axis('off')

    fig.suptitle("Neuron Response Curves", fontsize=14)
    fig.tight_layout()

    return fig


def plot_correlation_vs_reliability(correlations: np.ndarray, reliability: np.ndarray,
                                     show_labels: bool = True, top_n: int = 5):
    """
    Scatter plot of model correlation vs neuron reliability with ceiling and trend line.

    Args:
        correlations: Array of correlation values per neuron
        reliability: Array of reliability values per neuron
        show_labels: Whether to show neuron index labels
        top_n: Number of best/worst neurons to label (if show_labels=True)

    Returns:
        matplotlib.figure.Figure
    """
    reliability = np.array(reliability)
    correlations = np.array(correlations)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(reliability, correlations, s=80, alpha=0.7, edgecolors='black')

    # Ceiling line (y=x)
    max_val = max(np.nanmax(reliability), np.nanmax(correlations))
    min_val = min(np.nanmin(reliability), np.nanmin(correlations))
    lims = [min_val - 0.05, max_val + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Ceiling (y=x)')

    # Trend line
    mask = ~np.isnan(reliability) & ~np.isnan(correlations)
    if np.sum(mask) > 1:
        z = np.polyfit(reliability[mask], correlations[mask], 1)
        p = np.poly1d(z)
        x_range = np.linspace(np.min(reliability[mask]), np.max(reliability[mask]), 100)
        ax.plot(x_range, p(x_range), 'r--', linewidth=2,
                label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")

    # Labels for best/worst neurons
    if show_labels:
        top_by_corr = np.argsort(correlations)[-top_n:]
        bottom_by_corr = np.argsort(correlations)[:top_n]
        for idx in top_by_corr:
            ax.text(reliability[idx], correlations[idx], str(idx),
                    color='green', fontsize=9, ha='center', va='bottom')
        for idx in bottom_by_corr:
            ax.text(reliability[idx], correlations[idx], str(idx),
                    color='red', fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Reliability')
    ax.set_ylabel('Model Correlation')
    ax.set_title('Correlation vs. Reliability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_example_predictions(predictions: np.ndarray, targets: np.ndarray,
                              correlations: np.ndarray, reliability: np.ndarray = None,
                              foc: np.ndarray = None):
    """
    Plot actual vs predicted scatter for neurons at different performance percentiles.

    Selects 5 neurons: worst, 25th percentile, median, 75th percentile, best by correlation.

    Args:
        predictions: Model predictions, shape [num_samples, num_neurons]
        targets: Ground truth values, shape [num_samples, num_neurons]
        correlations: Correlation values per neuron
        reliability: Reliability values per neuron (optional)
        foc: Fraction of ceiling values per neuron (optional)

    Returns:
        matplotlib.figure.Figure
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    sorted_indices = np.argsort(correlations)
    n = len(sorted_indices)

    # Select neurons at different percentiles
    neurons_to_plot = [
        sorted_indices[0],                    # Worst (0th percentile)
        sorted_indices[n // 4],               # 25th percentile
        sorted_indices[n // 2],               # Median (50th percentile)
        sorted_indices[3 * n // 4],           # 75th percentile
        sorted_indices[-1]                    # Best (100th percentile)
    ]

    fig, axes = plt.subplots(len(neurons_to_plot), 1, figsize=(10, 3 * len(neurons_to_plot)))

    for i, n_idx in enumerate(neurons_to_plot):
        ax = axes[i]
        ax.scatter(targets[:, n_idx], predictions[:, n_idx], alpha=0.7, s=30)

        # Add identity line (y=x)
        min_val = min(np.min(targets[:, n_idx]), np.min(predictions[:, n_idx]))
        max_val = max(np.max(targets[:, n_idx]), np.max(predictions[:, n_idx]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        # Build title with metrics
        percentile = int(np.round(100 * i / (len(neurons_to_plot) - 1)))
        title = f'Neuron #{n_idx} ({percentile}th %ile) - Corr: {correlations[n_idx]:.4f}'

        if reliability is not None:
            rel_str = f'{reliability[n_idx]:.4f}' if not np.isnan(reliability[n_idx]) else 'N/A'
            title += f', Rel: {rel_str}'

        if foc is not None:
            foc_str = f'{foc[n_idx]:.4f}' if not np.isnan(foc[n_idx]) else 'N/A'
            title += f', FoC: {foc_str}'

        ax.set_title(title)
        ax.set_xlabel('Actual Response')
        ax.set_ylabel('Predicted Response')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_grid_predictions(predictions: np.ndarray, targets: np.ndarray,
                           correlations: np.ndarray, n_per_plot: int = 16):
    """
    Plot grid of actual vs predicted scatter plots for all neurons.

    Args:
        predictions: Model predictions, shape [num_samples, num_neurons]
        targets: Ground truth values, shape [num_samples, num_neurons]
        correlations: Correlation values per neuron
        n_per_plot: Maximum neurons per figure (default: 16)

    Returns:
        List of matplotlib.figure.Figure objects
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    n_neurons = targets.shape[1]
    figs = []

    for start_idx in range(0, n_neurons, n_per_plot):
        end_idx = min(start_idx + n_per_plot, n_neurons)
        n_to_plot = end_idx - start_idx
        grid_size = int(np.ceil(np.sqrt(n_to_plot)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), squeeze=False)
        axes = axes.flatten()

        for i, n_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            ax.scatter(targets[:, n_idx], predictions[:, n_idx], alpha=0.5, s=15)

            # Identity line
            lims = [
                min(np.min(targets[:, n_idx]), np.min(predictions[:, n_idx])),
                max(np.max(targets[:, n_idx]), np.max(predictions[:, n_idx]))
            ]
            ax.plot(lims, lims, 'r--', alpha=0.5)
            ax.set_title(f'Neuron {n_idx}: r={correlations[n_idx]:.2f}', fontsize=10)

            # Only show labels on outer edges
            if i % grid_size == 0:
                ax.set_ylabel('Predicted')
            if i >= (grid_size ** 2 - grid_size):
                ax.set_xlabel('Actual')

            ax.tick_params(axis='both', which='major', labelsize=8)

        # Hide unused subplots
        for j in range(n_to_plot, grid_size ** 2):
            axes[j].axis('off')

        plt.tight_layout()
        figs.append(fig)

    return figs


def plot_kernels(model, layer_idx: int = 0):
    """
    Plot convolutional kernels from the model.

    Args:
        model: KlindtCoreReadout2D model
        layer_idx: which conv layer to plot (default: 0)
    """
    kernels = model.core.conv_layers[layer_idx].weight.detach().cpu().numpy()
    # kernels shape: [out_channels, in_channels, H, W]
    
    n_kernels = kernels.shape[0]
    n_cols = min(4, n_kernels)
    n_rows = (n_kernels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_kernels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_kernels):
        ax = axes[i]
        kernel = kernels[i, 0]  # first input channel
        vmax = np.abs(kernel).max()
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Kernel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(n_kernels, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Convolutional Kernels')
    plt.tight_layout()
    return fig


def plot_kernels_dedicated(model, layer_idx: int = 0):
    """
    Plot convolutional kernels from KlindtCoreReadoutDedicatedONOFFMixed2D model,
    grouped by pathway (ON, OFF, Mixed).

    Shows kernels in 3 columns:
    - Column 1: ON kernels (positive polarity)
    - Column 2: OFF kernels (negative polarity)
    - Column 3: Mixed kernels (1 ON-like + 1 OFF-like)

    Args:
        model: KlindtCoreReadoutDedicatedONOFFMixed2D model
        layer_idx: which conv layer to plot (default: 0)

    Returns:
        matplotlib.figure.Figure
    """
    kernels = model.core.conv_layers[layer_idx].weight.detach().cpu().numpy()
    # kernels shape: [out_channels, in_channels, H, W]

    n_on = model.core.n_on_kernels
    n_off = model.core.n_off_kernels
    n_mixed = model.core.n_mixed_kernels

    # Determine grid layout: rows = max kernels per pathway, cols = 3 pathways
    n_rows = max(n_on, n_off, n_mixed)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    pathway_labels = ['ON Kernels', 'OFF Kernels', 'Mixed Kernels']
    pathway_counts = [n_on, n_off, n_mixed]
    pathway_starts = [0, n_on, n_on + n_off]

    # Find global vmax for consistent coloring
    vmax_global = np.abs(kernels[:, 0]).max()

    for col, (label, count, start) in enumerate(zip(pathway_labels, pathway_counts, pathway_starts)):
        for row in range(n_rows):
            ax = axes[row, col]
            if row < count:
                kernel_idx = start + row
                kernel = kernels[kernel_idx, 0]  # first input channel
                im = ax.imshow(kernel, cmap='RdBu_r', vmin=-vmax_global, vmax=vmax_global)
                ax.set_title(f'Kernel {kernel_idx}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.set_visible(False)

            if row == 0:
                ax.annotate(label, xy=(0.5, 1.15), xycoords='axes fraction',
                           fontsize=12, fontweight='bold', ha='center')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Convolutional Kernels by Pathway\n(ON: positive, OFF: negative, Mixed: 1 ON + 1 OFF)',
                 fontsize=14)
    plt.tight_layout()
    return fig


def plot_spatial_masks(model, neuron_indices: Optional[List[int]] = None):
    """
    Plot spatial masks from the readout layer.
    """
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()
    mask_size = model.readout.mask_size
    num_neurons = mask_weights.shape[1]
    
    if neuron_indices is None:
        neuron_indices = list(range(num_neurons))
    
    n_plots = len(neuron_indices)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]
        mask = mask_weights[:, neuron_idx].reshape(mask_size)
        im = ax.imshow(mask, cmap='gray')
        ax.set_title(f'Neuron {neuron_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Spatial Masks')
    plt.tight_layout()
    return fig


def plot_spatial_masks_per_channel(model, neurons_per_plot: int = 8):
    """
    Plot per-channel spatial masks from the KlindtReadoutPerChannel2D readout layer.

    Each neuron has one spatial mask per channel. This function creates multiple figures,
    each with up to `neurons_per_plot` neurons (rows) and all channels (columns).

    Args:
        model: Model with KlindtReadoutPerChannel2D readout (mask_weights shape: [H*W, C, N])
        neurons_per_plot: Number of neurons per figure (default: 8)

    Returns:
        List[matplotlib.figure.Figure]: List of figures, one per batch of neurons
    """
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, C, N]
    mask_size = model.readout.mask_size
    num_channels = mask_weights.shape[1]
    num_neurons = mask_weights.shape[2]

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)
        n_channels = num_channels

        fig, axes = plt.subplots(n_neurons, n_channels, figsize=(3*n_channels, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]
        if n_channels == 1:
            axes = axes[:, np.newaxis]

        for i, neuron_idx in enumerate(neuron_indices):
            for c in range(n_channels):
                ax = axes[i, c]
                mask = mask_weights[:, c, neuron_idx].reshape(mask_size)
                im = ax.imshow(mask, cmap='gray')
                if i == 0:
                    ax.set_title(f'Channel {c}')
                if c == 0:
                    ax.set_ylabel(f'Neuron {neuron_idx}')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(f'Per-Channel Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})')
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_spatial_masks_n(model, neurons_per_plot: int = 8):
    """
    Plot spatial masks from the KlindtReadoutNMasks2D readout layer.

    Each neuron has N spatial masks (e.g., 2 for center/surround). This function
    creates multiple figures, each with up to `neurons_per_plot` neurons (rows)
    and all masks (columns).

    Args:
        model: Model with KlindtReadoutNMasks2D readout (mask_weights shape: [H*W, M, N])
        neurons_per_plot: Number of neurons per figure (default: 8)

    Returns:
        List[matplotlib.figure.Figure]: List of figures, one per batch of neurons
    """
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, M, N]
    mask_size = model.readout.mask_size
    num_masks = mask_weights.shape[1]
    num_neurons = mask_weights.shape[2]

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        fig, axes = plt.subplots(n_neurons, num_masks, figsize=(3*num_masks, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]
        if num_masks == 1:
            axes = axes[:, np.newaxis]

        for i, neuron_idx in enumerate(neuron_indices):
            for m in range(num_masks):
                ax = axes[i, m]
                mask = mask_weights[:, m, neuron_idx].reshape(mask_size)
                im = ax.imshow(mask, cmap='gray')
                if i == 0:
                    ax.set_title(f'Mask {m}')
                if m == 0:
                    ax.set_ylabel(f'Neuron {neuron_idx}')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(f'Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})')
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_spatial_masks_onoff(model, neurons_per_plot: int = 8, shared_scale: bool = True):
    """
    Plot ON/OFF spatial masks from the KlindtReadoutONOFF2D readout layer.

    Each neuron has 2 spatial masks: ON mask (for ON kernels) and OFF mask (for OFF kernels).
    This function creates multiple figures, each with up to `neurons_per_plot` neurons (rows)
    and 2 columns (ON mask, OFF mask).

    Args:
        model: Model with KlindtReadoutONOFF2D readout (mask_weights shape: [H*W, 2, N])
        neurons_per_plot: Number of neurons per figure (default: 8)
        shared_scale: If True, use same vmin/vmax for both ON and OFF masks per neuron,
                      so empty/weak masks appear dark. If False, each mask auto-scales.

    Returns:
        List[matplotlib.figure.Figure]: List of figures, one per batch of neurons
    """
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, 2, N]
    mask_size = model.readout.mask_size
    num_neurons = mask_weights.shape[2]

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
            # Get both masks for this neuron
            on_mask = mask_weights[:, 0, neuron_idx].reshape(mask_size)
            off_mask = mask_weights[:, 1, neuron_idx].reshape(mask_size)

            if shared_scale:
                # Use shared scale: vmin=0, vmax=max of both masks
                # This way, empty/weak masks appear dark
                vmin = 0
                vmax = max(on_mask.max(), off_mask.max())
                if vmax == 0:
                    vmax = 1  # Avoid division by zero for completely empty masks
            else:
                vmin, vmax = None, None

            masks = [on_mask, off_mask]
            for m in range(2):
                ax = axes[i, m]
                if shared_scale:
                    im = ax.imshow(masks[m], cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(masks[m], cmap='gray')
                if i == 0:
                    ax.set_title(mask_labels[m], fontsize=12, fontweight='bold')
                if m == 0:
                    ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'ON/OFF Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_spatial_masks_onoff_mixed(model, neurons_per_plot: int = 8, shared_scale: bool = True):
    """
    Plot ON/OFF/Mixed spatial masks from the KlindtReadoutONOFFMixed2D readout layer.

    Each neuron has 3 spatial masks: ON mask, OFF mask, and Mixed mask.
    This function creates multiple figures, each with up to `neurons_per_plot` neurons (rows)
    and 3 columns (ON mask, OFF mask, Mixed mask).

    Args:
        model: Model with KlindtReadoutONOFFMixed2D readout (mask_weights shape: [H*W, 3, N])
        neurons_per_plot: Number of neurons per figure (default: 8)
        shared_scale: If True, use same vmin/vmax for all masks per neuron,
                      so empty/weak masks appear dark. If False, each mask auto-scales.

    Returns:
        List[matplotlib.figure.Figure]: List of figures, one per batch of neurons
    """
    mask_weights = model.readout.mask_weights.detach().cpu().numpy()  # [H*W, 3, N]
    mask_size = model.readout.mask_size
    num_neurons = mask_weights.shape[2]

    # Split neurons into batches
    neuron_batches = []
    for start in range(0, num_neurons, neurons_per_plot):
        end = min(start + neurons_per_plot, num_neurons)
        neuron_batches.append(list(range(start, end)))

    figures = []
    mask_labels = ['ON Mask', 'OFF Mask', 'Mixed Mask']

    for batch_idx, neuron_indices in enumerate(neuron_batches):
        n_neurons = len(neuron_indices)

        fig, axes = plt.subplots(n_neurons, 3, figsize=(9, 3*n_neurons))

        if n_neurons == 1:
            axes = axes[np.newaxis, :]

        for i, neuron_idx in enumerate(neuron_indices):
            # Get all three masks for this neuron
            on_mask = mask_weights[:, 0, neuron_idx].reshape(mask_size)
            off_mask = mask_weights[:, 1, neuron_idx].reshape(mask_size)
            mixed_mask = mask_weights[:, 2, neuron_idx].reshape(mask_size)

            if shared_scale:
                # Use shared scale: vmin=0, vmax=max of all masks
                vmin = 0
                vmax = max(on_mask.max(), off_mask.max(), mixed_mask.max())
                if vmax == 0:
                    vmax = 1  # Avoid division by zero for completely empty masks
            else:
                vmin, vmax = None, None

            masks = [on_mask, off_mask, mixed_mask]
            for m in range(3):
                ax = axes[i, m]
                if shared_scale:
                    im = ax.imshow(masks[m], cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(masks[m], cmap='gray')
                if i == 0:
                    ax.set_title(mask_labels[m], fontsize=12, fontweight='bold')
                if m == 0:
                    ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'ON/OFF/Mixed Spatial Masks (Neurons {neuron_indices[0]}-{neuron_indices[-1]})', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

    return figures


def _gaussian2D(shape, amp, x0, y0, sigma_x, sigma_y, angle):
    """Generate 2D Gaussian for RF ellipse overlay."""
    import math
    if sigma_x == 0:
        sigma_x = 0.001
    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]), int(shape[1]))
    x = np.linspace(0, shape[1], shape[1])
    y = np.linspace(0, shape[0], shape[0])
    X, Y = np.meshgrid(x, y)

    theta = math.pi * angle / 180
    a = (math.cos(theta)**2) / (2*sigma_x**2) + (math.sin(theta)**2) / (2*sigma_y**2)
    b = -(math.sin(2*theta)) / (4*sigma_x**2) + (math.sin(2*theta)) / (4*sigma_y**2)
    c = (math.sin(theta)**2) / (2*sigma_x**2) + (math.cos(theta)**2) / (2*sigma_y**2)

    return amp * np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))


def _overlay_rf_ellipse(ax, ellipse_coords, img_shape, rf_dim, scale_factor=1.0,
                        level_factor=0.35, line_width=2, color='yellow', alpha=0.6):
    """
    Overlay RF ellipse contour on an axis.

    Args:
        ax: matplotlib axis
        ellipse_coords: [amp, x0, y0, sigma_x, sigma_y, angle] in RF space
        img_shape: (H, W) of the image being plotted
        rf_dim: original RF dimension (e.g., 72 for STA)
        scale_factor: scaling from RF space to image space
        level_factor: contour level as fraction of max
        line_width: contour line width
        color: contour color
        alpha: contour transparency
    """
    if ellipse_coords is None or ellipse_coords[0] == 0:
        return

    img_h, img_w = img_shape

    # Convert RF coordinates to image space
    center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + img_w / 2
    center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + img_h / 2

    # Scale ellipse parameters
    scaled_ellipse = [
        ellipse_coords[0],
        center_x,
        center_y,
        ellipse_coords[3] * scale_factor,
        ellipse_coords[4] * scale_factor,
        ellipse_coords[5]
    ]

    # Generate Gaussian and overlay contour
    gaussian = _gaussian2D([img_h, img_w], *scaled_ellipse)
    ax.contour(
        np.abs(gaussian),
        levels=[level_factor * np.max(np.abs(gaussian))],
        colors=color,
        linewidths=line_width,
        alpha=alpha
    )


def plot_lsta_comparison_per_cell(
    images: np.ndarray,
    lsta_model: np.ndarray,
    lsta_exp: Optional[np.ndarray] = None,
    output_dir: str = None,
    neuron_indices: Optional[List[int]] = None,
    rf_fits: Optional[dict] = None,
    cell_ids: Optional[List[int]] = None,
    rf_dim: int = 72,
    cmap: str = 'RdBu_r',
    vmax_thresh: float = 0.5,
    expon_treat: int = 3,
    zoom: float = 1.0,
    target_size: Optional[int] = None,
):
    """
    Plot side-by-side comparison of original image, experimental LSTA, and model LSTA.
    Saves one figure per neuron in a subfolder. Optionally overlays RF ellipse.

    Args:
        images: Original images, shape [num_images, H, W] or [num_images, C, H, W]
        lsta_model: Model-predicted LSTA, shape [num_neurons, num_images, H, W]
        lsta_exp: Experimental LSTA (optional), shape [num_neurons, num_images, H, W]
        output_dir: Directory to save figures (will create 'lsta_per_cell' subfolder)
        neuron_indices: Which neurons to plot (default: all)
        rf_fits: RF fit data dict, loaded from sta_data_mix.pkl. Keys are cell IDs.
        cell_ids: List of cell IDs corresponding to neurons in the model
        rf_dim: Original RF/STA dimension (default: 72)
        cmap: Colormap for LSTA
        vmax_thresh: Threshold for color range as fraction of max
        expon_treat: Exponent for contrast enhancement (odd number)
        zoom: Zoom factor around RF center (1.0 = no zoom)
        target_size: Target size to resize all images/LSTAs to (default: image size)

    Returns:
        List of saved file paths
    """
    import os
    import cv2

    num_neurons, num_images = lsta_model.shape[:2]

    if neuron_indices is None:
        neuron_indices = list(range(num_neurons))

    # Handle image shape
    if images.ndim == 4:
        # [N, C, H, W] -> [N, H, W] (take first channel)
        images = images[:, 0, :, :]

    # Determine target size for all images (use image size by default)
    if target_size is None:
        target_size = images.shape[1]  # Use image height as target

    # Resize images if needed
    if images.shape[1] != target_size:
        resized_images = np.zeros((images.shape[0], target_size, target_size), dtype=images.dtype)
        for idx in range(images.shape[0]):
            resized_images[idx] = cv2.resize(images[idx], (target_size, target_size), interpolation=cv2.INTER_AREA)
        images = resized_images

    # Resize model LSTA if needed
    if lsta_model.shape[2] != target_size:
        resized_model = np.zeros((lsta_model.shape[0], lsta_model.shape[1], target_size, target_size), dtype=lsta_model.dtype)
        for n in range(lsta_model.shape[0]):
            for img in range(lsta_model.shape[1]):
                resized_model[n, img] = cv2.resize(lsta_model[n, img], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        lsta_model = resized_model

    # Resize experimental LSTA if needed
    has_exp = lsta_exp is not None
    if has_exp and lsta_exp.shape[2] != target_size:
        resized_exp = np.zeros((lsta_exp.shape[0], lsta_exp.shape[1], target_size, target_size), dtype=lsta_exp.dtype)
        for n in range(lsta_exp.shape[0]):
            for img in range(lsta_exp.shape[1]):
                resized_exp[n, img] = cv2.resize(lsta_exp[n, img], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        lsta_exp = resized_exp

    # Scale factor from RF space to target size
    scale_factor = target_size / rf_dim

    # Create output subfolder
    if output_dir is not None:
        lsta_dir = os.path.join(output_dir, 'lsta_per_cell')
        os.makedirs(lsta_dir, exist_ok=True)
    else:
        lsta_dir = None

    saved_paths = []
    has_rf = rf_fits is not None and cell_ids is not None

    for i, neuron_idx in enumerate(neuron_indices):
        # Get RF ellipse coordinates if available
        ellipse_coords = None
        cell_id = None
        if has_rf and i < len(cell_ids):
            cell_id = cell_ids[i]
            if cell_id in rf_fits:
                ellipse_coords = rf_fits[cell_id]['center_analyse']['EllipseCoor']

        # Layout: single row with grouped panels and spacing between groups
        # For each image: [Original, Exp LSTA (optional), Model LSTA]
        panels_per_image = 3 if has_exp else 2

        # Use GridSpec for better control over spacing
        from matplotlib.gridspec import GridSpec

        # Calculate figure width with space between groups
        panel_width = 2.5
        group_spacing = 0.8  # extra space between image groups
        fig_width = num_images * panels_per_image * panel_width + (num_images - 1) * group_spacing
        fig_height = 3.5

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create width ratios: panels within group are equal, but add extra space between groups
        width_ratios = []
        for img_idx in range(num_images):
            for p in range(panels_per_image):
                width_ratios.append(1)
            if img_idx < num_images - 1:
                width_ratios.append(0.3)  # spacer column

        total_cols = len(width_ratios)
        gs = GridSpec(1, total_cols, figure=fig, width_ratios=width_ratios, wspace=0.05)

        for img_idx in range(num_images):
            # Account for spacer columns
            base_col = img_idx * (panels_per_image + 1) if img_idx > 0 else 0
            if img_idx > 0:
                base_col = img_idx * panels_per_image + img_idx  # panels + spacers before this group

            # Panel 1: Original Image
            ax_img = fig.add_subplot(gs[0, base_col])
            img_data = images[img_idx]
            ax_img.imshow(img_data, cmap='gray')

            # Overlay RF ellipse on image
            if ellipse_coords is not None:
                _overlay_rf_ellipse(ax_img, ellipse_coords, img_data.shape, rf_dim,
                                   scale_factor=scale_factor, color='yellow')

                # Apply zoom around RF center
                if zoom > 1.0:
                    img_h, img_w = img_data.shape
                    center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + img_w / 2
                    center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + img_h / 2
                    zoom_half = (img_w / 2) / zoom
                    ax_img.set_xlim(center_x - zoom_half, center_x + zoom_half)
                    ax_img.set_ylim(center_y + zoom_half, center_y - zoom_half)

            # Add border around the axis
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f'Image {img_idx + 1}', fontsize=10)

            # Panel 2: Experimental LSTA (if provided)
            if has_exp:
                ax_exp = fig.add_subplot(gs[0, base_col + 1])
                lsta_e = lsta_exp[i, img_idx].copy()

                # Contrast enhancement
                lsta_e = np.sign(lsta_e) * (np.abs(lsta_e) ** expon_treat)
                vmax_e = np.max(np.abs(lsta_e)) * vmax_thresh
                if vmax_e > 0:
                    ax_exp.imshow(lsta_e, cmap=cmap, vmin=-vmax_e, vmax=vmax_e)
                else:
                    ax_exp.imshow(lsta_e, cmap=cmap)

                # Overlay RF ellipse on exp LSTA (same scale as image since resized)
                if ellipse_coords is not None:
                    _overlay_rf_ellipse(ax_exp, ellipse_coords, (target_size, target_size), rf_dim,
                                       scale_factor=scale_factor, color='yellow')

                    # Apply zoom (same as image)
                    if zoom > 1.0:
                        center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + target_size / 2
                        center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + target_size / 2
                        zoom_half = (target_size / 2) / zoom
                        ax_exp.set_xlim(center_x - zoom_half, center_x + zoom_half)
                        ax_exp.set_ylim(center_y + zoom_half, center_y - zoom_half)

                # Add border
                for spine in ax_exp.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
                ax_exp.set_xticks([])
                ax_exp.set_yticks([])
                ax_exp.set_title(f'Exp {img_idx + 1}', fontsize=10)

            # Panel 3 (or 2): Model LSTA
            model_col_offset = 2 if has_exp else 1
            ax_model = fig.add_subplot(gs[0, base_col + model_col_offset])
            lsta_m = lsta_model[i, img_idx].copy()

            # Contrast enhancement
            lsta_m = np.sign(lsta_m) * (np.abs(lsta_m) ** expon_treat)
            vmax_m = np.max(np.abs(lsta_m)) * vmax_thresh
            if vmax_m > 0:
                ax_model.imshow(lsta_m, cmap=cmap, vmin=-vmax_m, vmax=vmax_m)
            else:
                ax_model.imshow(lsta_m, cmap=cmap)

            # Overlay RF ellipse on model LSTA (same scale as image since resized)
            if ellipse_coords is not None:
                _overlay_rf_ellipse(ax_model, ellipse_coords, (target_size, target_size), rf_dim,
                                   scale_factor=scale_factor, color='yellow')

                # Apply zoom (same as image)
                if zoom > 1.0:
                    center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + target_size / 2
                    center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + target_size / 2
                    zoom_half = (target_size / 2) / zoom
                    ax_model.set_xlim(center_x - zoom_half, center_x + zoom_half)
                    ax_model.set_ylim(center_y + zoom_half, center_y - zoom_half)

            # Add border
            for spine in ax_model.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
            ax_model.set_xticks([])
            ax_model.set_yticks([])
            ax_model.set_title(f'Model {img_idx + 1}', fontsize=10)

        # Title with cell ID if available
        title = f'Neuron {neuron_idx}'
        if cell_id is not None:
            title += f' (Cell {cell_id})'
        title += ' - LSTA Comparison'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        if lsta_dir is not None:
            filename = f'neuron_{neuron_idx:03d}'
            if cell_id is not None:
                filename += f'_cell_{cell_id}'
            filename += '_lsta.png'
            save_path = os.path.join(lsta_dir, filename)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            plt.close(fig)
        else:
            plt.show()

    return saved_paths


# ============================================================
# Polarity helpers (shared with analyze_spatial_masks.py)
# ============================================================

def _pathway_polarity(kernel_centers, weights_for_neuron, threshold=0.005):
    """Compute per-kernel contributions and classify pathway polarity.

    Args:
        kernel_centers: center pixel values for each kernel in this pathway
        weights_for_neuron: readout weights for each kernel for one neuron
        threshold: minimum |readout_weight| to count a kernel as used

    Returns:
        on_part: sum of positive center*weight products
        off_part: sum of |negative center*weight products|
        has_on: whether any used kernel contributes ON
        has_off: whether any used kernel contributes OFF
    """
    on_part = 0.0
    off_part = 0.0
    for k in range(len(kernel_centers)):
        if abs(weights_for_neuron[k]) < threshold:
            continue
        cw = kernel_centers[k] * weights_for_neuron[k]
        if cw >= 0:
            on_part += cw
        else:
            off_part += abs(cw)
    has_on = on_part > 0
    has_off = off_part > 0
    return on_part, off_part, has_on, has_off


def _pathway_cmap_and_color(has_on, has_off):
    """Return (colormap, text_color) based on pathway polarity mix."""
    if has_on and has_off:
        return 'Greens', 'darkgreen'
    elif has_on:
        return 'Reds', 'darkred'
    else:
        return 'Blues', 'darkblue'


# ============================================================
# Effective Receptive Field
# ============================================================

def compute_effective_rf(mask_2d, pathway_kernels, pathway_weights):
    """Compute effective RF by convolving spatial mask with all kernels in a pathway.

    The effective linear RF combines "where" (spatial mask) and "what" (kernels)
    into a single image-space picture.  Each kernel contributes proportionally
    to its readout weight, giving the true linear RF for the pathway:

        EffRF = Σ_k  w_k × conv2d(mask, kernel_k)

    This naturally accounts for off-center kernels whose gradient-based analysis
    may be misleading.

    Args:
        mask_2d: [H_mask, W_mask] spatial mask for one pathway/neuron.
        pathway_kernels: [n_kernels, kH, kW] all kernels for this pathway.
        pathway_weights: [n_kernels] readout weights for each kernel.

    Returns:
        eff_rf: [H_mask + kH - 1, W_mask + kW - 1] effective RF in image space.
    """
    from scipy.signal import convolve2d
    result = None
    for k in range(len(pathway_kernels)):
        contribution = pathway_weights[k] * convolve2d(mask_2d, pathway_kernels[k], mode='full')
        if result is None:
            result = contribution
        else:
            result += contribution
    return result


# ============================================================
# Cell ID Card
# ============================================================

def plot_cell_id_cards(
    # Kernels
    kernels, n_on, n_off, n_mixed,
    # Spatial masks (per neuron)
    on_masks, off_masks, mixed_masks,
    # Kernel polarity info
    on_weights, off_weights, mixed_weights,
    on_kernel_centers, off_kernel_centers, mixed_kernel_centers,
    # Multi-mask LSTA data (dicts keyed by mask_idx)
    images_all,
    lsta_model_all,
    lsta_exp_all=None,
    mask_labels=None,
    # Mask batch info (for exp LSTA masking)
    masked_cells=None,
    all_RFs=None,
    accepted_lstas=None,
    # Response data
    predictions=None, targets=None,
    correlations=None, reliability=None,
    # Metadata
    cell_ids=None, cell_type=None,
    # RF overlay
    rf_fits=None, rf_dim=72,
    # LSTA display
    cmap='RdBu_r', vmax_thresh=0.5, expon_treat=3,
    zoom=1.0, target_size=432,
    smooth_sigma=5, hide_non_accepted=True,
    # Output
    output_dir=None,
    # Mask/polarity settings
    noise_ratio_threshold=0.1, polarity_threshold=0.005,
):
    """
    Generate a single-PNG summary ("ID card") for each neuron.

    Layout (nested GridSpecs):
      Section 0: Kernels (ON/OFF/Mixed) + metrics text panel
      Section 1: Spatial masks (ON, OFF, Mixed pathway, Combined polarity, Contours)
      Section 2: Effective RF (strongest-kernel convolution with mask, per pathway + combined)
      Section 3: Multi-mask LSTA grid (rows = mask conditions, cols = [Img, Exp, Model] x 4)
      Section 4: Response time-series

    Args:
        kernels: [n_total, kH, kW] conv kernel weights (first input channel)
        n_on, n_off, n_mixed: number of kernels per pathway
        on_masks, off_masks, mixed_masks: [H, W, N] spatial mask arrays
        on_weights, off_weights, mixed_weights: [n_k, N] readout weights per pathway
        on_kernel_centers, off_kernel_centers, mixed_kernel_centers: kernel center values
        images_all: dict mask_idx -> [num_images, 1, H, W] stimulus images per condition
        lsta_model_all: dict mask_idx -> [N, num_images, H, W] model LSTA per condition
        lsta_exp_all: dict mask_idx -> [N, num_images, H, W] experimental LSTA (optional)
        mask_labels: list of str, e.g. ['Original', 'Mask 1', ...]
        masked_cells: list of cell ID batches per mask condition (for exp LSTA masking)
        all_RFs: list of (cell_id, ellipse_coords) for batch masking
        accepted_lstas: dict cell_id -> {mask_idx -> {img_idx -> bool}}
        predictions: [T, N] model predictions on test set
        targets: [T, N] ground truth responses on test set
        correlations: [N] per-neuron correlation
        reliability: [N] per-neuron reliability
        cell_ids: list of cell IDs
        cell_type: string label for cell type
        rf_fits: RF fit data dict for ellipse overlay
        rf_dim: original RF dimension (default 72)
        cmap: colormap for LSTA
        vmax_thresh: LSTA color range threshold
        expon_treat: LSTA contrast enhancement exponent
        zoom: zoom factor around RF center
        target_size: display size for LSTA panels (default 432)
        smooth_sigma: Gaussian smoothing sigma for LSTA (default 5)
        hide_non_accepted: hide non-accepted LSTA panels (default True)
        output_dir: directory to save PNGs
        noise_ratio_threshold: threshold for mask noise detection
        polarity_threshold: threshold for kernel polarity detection

    Returns:
        List of saved file paths
    """
    import os
    import cv2
    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import gaussian_filter
    from .lsta import apply_batch_mask_to_lsta

    has_on = n_on > 0
    has_off = n_off > 0
    has_mixed = n_mixed > 0
    # Build list of active pathways for mask/eff-RF sections
    active_pathways = []
    if has_on:
        active_pathways.append('ON')
    if has_off:
        active_pathways.append('OFF')
    if has_mixed:
        active_pathways.append('Mixed')

    num_neurons = on_masks.shape[2]
    H_mask, W_mask = on_masks.shape[:2]
    num_masks = len(images_all)
    num_images = images_all[0].shape[0]
    panels_per_image = 3  # Img, Exp, Model
    has_exp = lsta_exp_all is not None
    has_response = predictions is not None and targets is not None

    if mask_labels is None:
        mask_labels = [f'Mask {i}' if i > 0 else 'Original' for i in range(num_masks)]

    scale_factor = target_size / rf_dim

    # Output directory
    if output_dir is not None:
        card_dir = os.path.join(output_dir, 'cell_id_cards')
        os.makedirs(card_dir, exist_ok=True)
    else:
        card_dir = None

    saved_paths = []

    for neuron_idx in range(num_neurons):
        cell_id = cell_ids[neuron_idx] if cell_ids is not None else None

        # ---------- Extract per-neuron data ----------
        on_mask = on_masks[:, :, neuron_idx]
        off_mask = off_masks[:, :, neuron_idx]
        mixed_mask = mixed_masks[:, :, neuron_idx]

        # Mask signal detection
        on_max = on_mask.max()
        off_max = off_mask.max()
        mixed_max = mixed_mask.max()
        strongest_max = max(on_max, off_max, mixed_max)
        if strongest_max == 0:
            on_is_signal = off_is_signal = mixed_is_signal = False
        else:
            on_is_signal = on_max >= noise_ratio_threshold * strongest_max
            off_is_signal = off_max >= noise_ratio_threshold * strongest_max
            mixed_is_signal = mixed_max >= noise_ratio_threshold * strongest_max
        vmax_mask = strongest_max if strongest_max > 0 else 1

        # Pathway polarity
        on_on, on_off, on_has_on, on_has_off = _pathway_polarity(
            on_kernel_centers, on_weights[:, neuron_idx], polarity_threshold)
        off_on, off_off, off_has_on, off_has_off = _pathway_polarity(
            off_kernel_centers, off_weights[:, neuron_idx], polarity_threshold)
        mix_on, mix_off, mix_has_on, mix_has_off = _pathway_polarity(
            mixed_kernel_centers, mixed_weights[:, neuron_idx], polarity_threshold)

        # RF ellipse
        ellipse_coords = None
        if rf_fits is not None and cell_id is not None and cell_id in rf_fits:
            ellipse_coords = rf_fits[cell_id]['center_analyse']['EllipseCoor']

        # Metrics
        corr = correlations[neuron_idx] if correlations is not None else None
        rel = reliability[neuron_idx] if reliability is not None else None
        foc = (corr / rel) if (corr is not None and rel is not None and rel > 0) else None

        # ---------- Create figure with nested GridSpecs ----------
        panel_width = 2.0
        panel_height = 2.0
        fig_width = (num_images * panels_per_image * panel_width
                     + (num_images - 1) * 0.8 + 2)
        kernel_h = 3.0
        mask_h = 3.5
        eff_rf_h = 3.5
        lsta_h = num_masks * panel_height + 1
        response_h = 2.0 if has_response else 0
        fig_height = kernel_h + mask_h + eff_rf_h + lsta_h + response_h + 2

        fig = plt.figure(figsize=(fig_width, fig_height))

        n_outer_rows = 4 + (1 if has_response else 0)
        height_ratios = [kernel_h, mask_h, eff_rf_h, lsta_h]
        if has_response:
            height_ratios.append(response_h)

        outer_gs = GridSpec(n_outer_rows, 1, figure=fig,
                            height_ratios=height_ratios, hspace=0.25)

        # ========== Section 0: Kernels + Metrics ==========
        n_total_kernels = n_on + n_off + n_mixed
        n_kernel_grid_cols = n_total_kernels + 1  # all kernels + metrics panel
        gs_k = outer_gs[0].subgridspec(1, n_kernel_grid_cols, wspace=0.3)
        vmax_kernel = np.abs(kernels).max()

        pathway_labels = []
        for k in range(n_on):
            pathway_labels.append(f'ON K{k}')
        for k in range(n_off):
            pathway_labels.append(f'OFF K{k}')
        for k in range(n_mixed):
            pathway_labels.append(f'Mix K{k}')

        for k_idx in range(n_total_kernels):
            ax = fig.add_subplot(gs_k[0, k_idx])
            im = ax.imshow(kernels[k_idx], cmap='RdBu_r', vmin=-vmax_kernel, vmax=vmax_kernel)
            ax.set_title(pathway_labels[k_idx], fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_metrics = fig.add_subplot(gs_k[0, n_total_kernels])
        ax_metrics.axis('off')
        metrics_text = ""
        if corr is not None:
            metrics_text += f"Correlation: {corr:.4f}\n"
        if rel is not None:
            metrics_text += f"Reliability: {rel:.4f}\n"
        if foc is not None:
            metrics_text += f"FoC: {foc:.4f}\n"
        ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=14, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        # ========== Section 1: Spatial Masks ==========
        # Dynamic columns: one per active pathway + Combined + Contours
        n_mask_cols = len(active_pathways) + 2  # pathways + combined + contours
        gs_m = outer_gs[1].subgridspec(1, n_mask_cols, wspace=0.2)

        # Per-pathway mask data (only active pathways)
        mask_pw_data = []
        if has_on:
            mask_pw_data.append(('ON', on_mask, on_has_on, on_has_off, on_is_signal, on_max,
                                 on_kernel_centers, on_weights[:, neuron_idx]))
        if has_off:
            mask_pw_data.append(('OFF', off_mask, off_has_on, off_has_off, off_is_signal, off_max,
                                 off_kernel_centers, off_weights[:, neuron_idx]))
        if has_mixed:
            mask_pw_data.append(('Mixed', mixed_mask, mix_has_on, mix_has_off, mixed_is_signal, mixed_max,
                                 mixed_kernel_centers, mixed_weights[:, neuron_idx]))

        for col_idx, (pw_name, pw_mask, pw_has_on, pw_has_off, pw_is_signal, pw_max,
                       pw_centers, pw_w) in enumerate(mask_pw_data):
            ax = fig.add_subplot(gs_m[0, col_idx])
            cmap_pw, _ = _pathway_cmap_and_color(pw_has_on, pw_has_off)
            ax.imshow(pw_mask, cmap=cmap_pw, vmin=0, vmax=vmax_mask)
            ax.set_title(f'{pw_name} Pathway', fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

        # Combined polarity (RGB: R=ON, G=overlap, B=OFF)
        combined_col = len(active_pathways)
        ax = fig.add_subplot(gs_m[0, combined_col])
        on_channel = np.zeros((H_mask, W_mask))
        off_channel = np.zeros((H_mask, W_mask))
        for (pw_name, pw_mask, pw_has_on, pw_has_off, pw_is_signal, pw_max,
             pw_centers, pw_w) in mask_pw_data:
            if not pw_is_signal:
                continue
            for k in range(len(pw_centers)):
                if abs(pw_w[k]) < polarity_threshold:
                    continue
                contrib = pw_centers[k] * pw_w[k]
                if contrib >= 0:
                    on_channel += pw_mask * contrib
                else:
                    off_channel += pw_mask * abs(contrib)

        max_val = max(on_channel.max(), off_channel.max(), 1e-8)
        on_norm = on_channel / max_val
        off_norm = off_channel / max_val
        on_peak = on_channel.max()
        off_peak = off_channel.max()
        on_present = on_channel > 0.05 * on_peak if on_peak > 0 else np.full((H_mask, W_mask), False)
        off_present = off_channel > 0.05 * off_peak if off_peak > 0 else np.full((H_mask, W_mask), False)
        both = on_present & off_present
        on_only_px = on_present & ~off_present
        off_only_px = off_present & ~on_present
        rgb = np.ones((H_mask, W_mask, 3))  # white background
        rgb[on_only_px] = np.stack([np.ones_like(on_norm[on_only_px]),
                                     1 - on_norm[on_only_px],
                                     1 - on_norm[on_only_px]], axis=-1)
        rgb[off_only_px] = np.stack([1 - off_norm[off_only_px],
                                      1 - off_norm[off_only_px],
                                      np.ones_like(off_norm[off_only_px])], axis=-1)
        rgb[both] = np.stack([1 - np.maximum(on_norm[both], off_norm[both]),
                               np.ones_like(on_norm[both]),
                               1 - np.maximum(on_norm[both], off_norm[both])], axis=-1)
        ax.imshow(rgb)
        ax.set_title('Combined Polarity\n(R=ON, G=ON+OFF, B=OFF)', fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # Contour overlay
        contour_col = len(active_pathways) + 1
        ax = fig.add_subplot(gs_m[0, contour_col])
        ax.imshow(np.ones((H_mask, W_mask, 3)) * 0.9)
        for (pw_name, pw_mask, pw_has_on, pw_has_off, pw_is_signal, pw_max,
             pw_centers, pw_w) in mask_pw_data:
            if not pw_is_signal or pw_max == 0:
                continue
            if pw_has_on and pw_has_off:
                color = 'green'
            elif pw_has_on:
                color = 'red'
            else:
                color = 'blue'
            levels = [pw_max * t for t in [0.2, 0.5, 0.8]]
            ax.contour(pw_mask, levels=levels, colors=color, linewidths=1.5)
            ax.contourf(pw_mask, levels=[levels[1], pw_max], colors=[color], alpha=0.3)
        ax.set_title('Contours', fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # ========== Section 2: Effective RF ==========
        n_eff_cols = len(active_pathways) + 2  # pathways + combined + contours
        gs_eff = outer_gs[2].subgridspec(1, n_eff_cols, wspace=0.2)

        eff_rfs = []  # collect per-pathway effective RFs for combined
        eff_pw_configs = []
        if has_on:
            eff_pw_configs.append(('ON', on_mask, kernels[:n_on], on_weights[:, neuron_idx],
                                   on_is_signal))
        if has_off:
            eff_pw_configs.append(('OFF', off_mask, kernels[n_on:n_on + n_off],
                                   off_weights[:, neuron_idx], off_is_signal))
        if has_mixed:
            eff_pw_configs.append(('Mixed', mixed_mask, kernels[n_on + n_off:],
                                   mixed_weights[:, neuron_idx], mixed_is_signal))

        for pw_idx, (pw_name, pw_mask, pw_kernels, pw_w, pw_is_signal) in enumerate(eff_pw_configs):
            ax = fig.add_subplot(gs_eff[0, pw_idx])
            if pw_is_signal and len(pw_kernels) > 0 and np.any(np.abs(pw_w) >= polarity_threshold):
                eff = compute_effective_rf(pw_mask, pw_kernels, pw_w)
                eff_rfs.append(eff)
                vabs = np.abs(eff).max()
                if vabs > 0:
                    ax.imshow(eff, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
                else:
                    ax.imshow(eff, cmap='RdBu_r')
                n_active = int(np.sum(np.abs(pw_w) >= polarity_threshold))
                ax.set_title(f'{pw_name} Eff. RF\n({n_active} kernels)',
                             fontsize=9, fontweight='bold')
            else:
                ax.imshow(np.zeros((H_mask, W_mask)), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{pw_name} Eff. RF\n(no signal)', fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

        # Combined effective RF — accumulate ON/OFF separately per pathway
        eff_combined_col = len(active_pathways)
        ax = fig.add_subplot(gs_eff[0, eff_combined_col])
        if eff_rfs:
            # Pad all to same shape
            max_h = max(e.shape[0] for e in eff_rfs)
            max_w = max(e.shape[1] for e in eff_rfs)
            # Accumulate positive (ON) and negative (OFF) parts independently
            on_channel_eff = np.zeros((max_h, max_w))
            off_channel_eff = np.zeros((max_h, max_w))
            for e in eff_rfs:
                pad_h = (max_h - e.shape[0]) // 2
                pad_w = (max_w - e.shape[1]) // 2
                e_padded = np.zeros((max_h, max_w))
                e_padded[pad_h:pad_h + e.shape[0],
                         pad_w:pad_w + e.shape[1]] = e
                on_channel_eff += np.maximum(e_padded, 0)
                off_channel_eff += np.maximum(-e_padded, 0)

            max_val_eff = max(on_channel_eff.max(), off_channel_eff.max(), 1e-8)
            on_norm_eff = on_channel_eff / max_val_eff
            off_norm_eff = off_channel_eff / max_val_eff
            on_peak_eff = on_channel_eff.max()
            off_peak_eff = off_channel_eff.max()
            on_present_eff = on_channel_eff > 0.05 * on_peak_eff if on_peak_eff > 0 else np.full((max_h, max_w), False)
            off_present_eff = off_channel_eff > 0.05 * off_peak_eff if off_peak_eff > 0 else np.full((max_h, max_w), False)
            both_eff = on_present_eff & off_present_eff
            on_only_eff = on_present_eff & ~off_present_eff
            off_only_eff = off_present_eff & ~on_present_eff
            rgb_eff = np.ones((max_h, max_w, 3))  # white background
            rgb_eff[on_only_eff] = np.stack([np.ones_like(on_norm_eff[on_only_eff]),
                                              1 - on_norm_eff[on_only_eff],
                                              1 - on_norm_eff[on_only_eff]], axis=-1)
            rgb_eff[off_only_eff] = np.stack([1 - off_norm_eff[off_only_eff],
                                               1 - off_norm_eff[off_only_eff],
                                               np.ones_like(off_norm_eff[off_only_eff])], axis=-1)
            rgb_eff[both_eff] = np.stack([1 - np.maximum(on_norm_eff[both_eff], off_norm_eff[both_eff]),
                                           np.ones_like(on_norm_eff[both_eff]),
                                           1 - np.maximum(on_norm_eff[both_eff], off_norm_eff[both_eff])], axis=-1)
            ax.imshow(rgb_eff)
        else:
            ax.imshow(np.ones((H_mask, W_mask, 3)))
        ax.set_title('Combined Eff. RF\n(R=ON, G=ON+OFF, B=OFF)', fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # Combined effective RF contour
        eff_contour_col = len(active_pathways) + 1
        ax = fig.add_subplot(gs_eff[0, eff_contour_col])
        if eff_rfs:
            ax.imshow(np.ones((max_h, max_w, 3)) * 0.9)
            if on_peak_eff > 0:
                levels = [on_peak_eff * t for t in [0.2, 0.5, 0.8]]
                ax.contour(on_channel_eff, levels=levels, colors='red', linewidths=1.5)
                ax.contourf(on_channel_eff, levels=[levels[1], on_peak_eff], colors=['red'], alpha=0.3)
            if off_peak_eff > 0:
                levels = [off_peak_eff * t for t in [0.2, 0.5, 0.8]]
                ax.contour(off_channel_eff, levels=levels, colors='blue', linewidths=1.5)
                ax.contourf(off_channel_eff, levels=[levels[1], off_peak_eff], colors=['blue'], alpha=0.3)
        else:
            ax.imshow(np.ones((H_mask, W_mask, 3)) * 0.9)
        ax.set_title('Eff. RF Contours\n(R=ON, B=OFF)', fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # ========== Section 3: Multi-mask LSTA Grid ==========
        # Same layout as plot_lsta_all_masks(): label col + [Img,Exp,Model,spacer]*N
        lsta_width_ratios = [0.3]  # label column
        for img_idx in range(num_images):
            for _ in range(panels_per_image):
                lsta_width_ratios.append(1)
            if img_idx < num_images - 1:
                lsta_width_ratios.append(0.3)  # spacer between image groups

        gs_lsta = outer_gs[3].subgridspec(
            num_masks, len(lsta_width_ratios),
            width_ratios=lsta_width_ratios,
            wspace=0.05, hspace=0.15,
        )

        def _id_apply_zoom(ax_z):
            """Apply zoom around RF center."""
            if ellipse_coords is not None and zoom > 1.0:
                cx = (ellipse_coords[1] - rf_dim / 2) * scale_factor + target_size / 2
                cy = (ellipse_coords[2] - rf_dim / 2) * scale_factor + target_size / 2
                zh = (target_size / 2) / zoom
                ax_z.set_xlim(cx - zh, cx + zh)
                ax_z.set_ylim(cy + zh, cy - zh)

        for mask_idx in range(num_masks):
            images = images_all[mask_idx]
            lsta_model = lsta_model_all[mask_idx]
            lsta_exp = lsta_exp_all[mask_idx] if has_exp else None

            # Row label
            ax_label = fig.add_subplot(gs_lsta[mask_idx, 0])
            ax_label.text(0.5, 0.5, mask_labels[mask_idx], rotation=90,
                         va='center', ha='center', fontsize=10, fontweight='bold')
            ax_label.axis('off')

            for img_idx in range(num_images):
                # label col + (panels + spacer) per preceding group
                base_col = 1 + img_idx * (panels_per_image + 1)

                # Check acceptance
                is_accepted = False
                if accepted_lstas is not None and cell_id is not None:
                    if cell_id in accepted_lstas:
                        if mask_idx in accepted_lstas[cell_id]:
                            is_accepted = accepted_lstas[cell_id][mask_idx].get(img_idx + 1, False)
                is_original = (mask_idx == 0)
                show_lsta = (is_accepted or is_original) if hide_non_accepted else True

                # --- Image panel ---
                ax_img = fig.add_subplot(gs_lsta[mask_idx, base_col])
                img_data = images[img_idx, 0]  # [H, W]
                if img_data.shape[0] != target_size:
                    img_display = cv2.resize(img_data, (target_size, target_size),
                                             interpolation=cv2.INTER_AREA)
                else:
                    img_display = img_data
                ax_img.imshow(img_display, cmap='gray')
                if ellipse_coords is not None:
                    _overlay_rf_ellipse(ax_img, ellipse_coords, (target_size, target_size),
                                       rf_dim, scale_factor=scale_factor, color='red')
                    _id_apply_zoom(ax_img)
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
                ax_img.set_xticks([]); ax_img.set_yticks([])
                if mask_idx == 0:
                    ax_img.set_title(f'Img {img_idx + 1}', fontsize=9)

                # --- Experimental LSTA panel ---
                ax_exp = fig.add_subplot(gs_lsta[mask_idx, base_col + 1])
                if has_exp and show_lsta:
                    lsta_e = lsta_exp[neuron_idx, img_idx].copy()
                    # Apply batch mask for non-original conditions
                    if mask_idx > 0 and masked_cells is not None and all_RFs is not None:
                        batch = masked_cells[mask_idx - 1]
                        lsta_e = apply_batch_mask_to_lsta(
                            lsta_e, batch, all_RFs, bg_value=0.0, scale=6)
                    if lsta_e.shape[0] != target_size:
                        lsta_e = cv2.resize(lsta_e, (target_size, target_size),
                                            interpolation=cv2.INTER_AREA)
                    lsta_e = np.sign(lsta_e) * (np.abs(lsta_e) ** expon_treat)
                    if smooth_sigma and smooth_sigma > 0:
                        lsta_e = gaussian_filter(lsta_e, sigma=smooth_sigma)
                    vmax_e = np.max(np.abs(lsta_e)) * vmax_thresh
                    if vmax_e > 0:
                        ax_exp.imshow(lsta_e, cmap=cmap, vmin=-vmax_e, vmax=vmax_e)
                    else:
                        ax_exp.imshow(lsta_e, cmap=cmap)
                    if ellipse_coords is not None:
                        _overlay_rf_ellipse(ax_exp, ellipse_coords, (target_size, target_size),
                                           rf_dim, scale_factor=scale_factor, color='red')
                        _id_apply_zoom(ax_exp)
                    border_color = 'green' if is_accepted else 'black'
                    border_width = 3.0 if is_accepted else 1.5
                elif has_exp:
                    ax_exp.imshow(np.ones((target_size, target_size)),
                                  cmap='gray', vmin=0, vmax=1)
                    border_color = 'lightgray'
                    border_width = 1.5
                else:
                    ax_exp.imshow(np.ones((10, 10)), cmap='gray', vmin=0, vmax=1)
                    border_color = 'lightgray'
                    border_width = 1.5
                for spine in ax_exp.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(border_width)
                    spine.set_color(border_color)
                ax_exp.set_xticks([]); ax_exp.set_yticks([])
                if mask_idx == 0:
                    ax_exp.set_title(f'Exp {img_idx + 1}', fontsize=9)

                # --- Model LSTA panel ---
                ax_model = fig.add_subplot(gs_lsta[mask_idx, base_col + 2])
                if show_lsta:
                    lsta_m = lsta_model[neuron_idx, img_idx].copy()
                    if lsta_m.shape[0] != target_size:
                        lsta_m = cv2.resize(lsta_m, (target_size, target_size),
                                            interpolation=cv2.INTER_LINEAR)
                    lsta_m = np.sign(lsta_m) * (np.abs(lsta_m) ** expon_treat)
                    if smooth_sigma and smooth_sigma > 0:
                        lsta_m = gaussian_filter(lsta_m, sigma=smooth_sigma)
                    vmax_m = np.max(np.abs(lsta_m)) * vmax_thresh
                    if vmax_m > 0:
                        ax_model.imshow(lsta_m, cmap=cmap, vmin=-vmax_m, vmax=vmax_m)
                    else:
                        ax_model.imshow(lsta_m, cmap=cmap)
                    if ellipse_coords is not None:
                        _overlay_rf_ellipse(ax_model, ellipse_coords, (target_size, target_size),
                                           rf_dim, scale_factor=scale_factor, color='red')
                        _id_apply_zoom(ax_model)
                    model_border_color = 'green' if is_accepted else 'black'
                    model_border_width = 3.0 if is_accepted else 1.5
                else:
                    ax_model.imshow(np.ones((target_size, target_size)),
                                    cmap='gray', vmin=0, vmax=1)
                    model_border_color = 'lightgray'
                    model_border_width = 1.5
                for spine in ax_model.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(model_border_width)
                    spine.set_color(model_border_color)
                ax_model.set_xticks([]); ax_model.set_yticks([])
                if mask_idx == 0:
                    ax_model.set_title(f'Model {img_idx + 1}', fontsize=9)

        # ========== Section 3: Response ==========
        if has_response:
            gs_r = outer_gs[4].subgridspec(1, 1)
            ax_ts = fig.add_subplot(gs_r[0, 0])
            ax_ts.plot(targets[:, neuron_idx], label='Target', alpha=0.8, linewidth=1)
            ax_ts.plot(predictions[:, neuron_idx], label='Predicted', alpha=0.8, linewidth=1)
            title_str = 'Response Time Series'
            if corr is not None:
                title_str += f' (r={corr:.3f})'
            ax_ts.set_title(title_str, fontsize=9)
            ax_ts.set_xlabel('Sample', fontsize=8)
            ax_ts.set_ylabel('Response', fontsize=8)
            ax_ts.legend(fontsize=7)
            ax_ts.tick_params(labelsize=7)
            ax_ts.grid(True, alpha=0.3)

        # ========== Title ==========
        title = f'Cell ID Card'
        if cell_id is not None:
            title += f' - Cell {cell_id}'
        if cell_type is not None:
            title += f' | {cell_type}'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

        # ========== Save ==========
        if card_dir is not None:
            filename = f'cell_{cell_id}.png' if cell_id is not None else f'neuron_{neuron_idx:03d}.png'
            save_path = os.path.join(card_dir, filename)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            plt.close(fig)
        else:
            plt.show()

    if card_dir is not None:
        print(f"Saved {len(saved_paths)} cell ID cards to: {card_dir}")

    return saved_paths