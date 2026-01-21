import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


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


def plot_correlation_vs_reliability(correlations: np.ndarray, reliability: np.ndarray):
    """
    Scatter plot of model correlation vs neuron reliability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(reliability, correlations, s=80, alpha=0.7, edgecolors='black')
    
    for i, (rel, corr) in enumerate(zip(reliability, correlations)):
        ax.annotate(str(i), (rel, corr), fontsize=9, ha='center', va='bottom')
    
    lims = [min(reliability.min(), correlations.min()) - 0.1,
            max(reliability.max(), correlations.max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    
    ax.set_xlabel('Reliability')
    ax.set_ylabel('Model Correlation')
    ax.set_title('Model Performance vs Neuron Reliability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


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

        # Determine number of columns: 2 (image + model) or 3 (image + exp + model)
        n_cols = 3 if has_exp else 2

        fig, axes = plt.subplots(num_images, n_cols, figsize=(4 * n_cols, 4 * num_images))

        if num_images == 1:
            axes = axes[np.newaxis, :]

        for img_idx in range(num_images):
            col = 0

            # Column 1: Original Image
            ax_img = axes[img_idx, col]
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

            ax_img.axis('off')
            if img_idx == 0:
                ax_img.set_title('Original Image', fontsize=12, fontweight='bold')
            col += 1

            # Column 2: Experimental LSTA (if provided)
            if has_exp:
                ax_exp = axes[img_idx, col]
                lsta_e = lsta_exp[i, img_idx].copy()

                # Contrast enhancement
                lsta_e = np.sign(lsta_e) * (np.abs(lsta_e) ** expon_treat)
                vmax_e = np.max(np.abs(lsta_e)) * vmax_thresh
                if vmax_e > 0:
                    im_exp = ax_exp.imshow(lsta_e, cmap=cmap, vmin=-vmax_e, vmax=vmax_e)
                    plt.colorbar(im_exp, ax=ax_exp, fraction=0.046)
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

                ax_exp.axis('off')
                if img_idx == 0:
                    ax_exp.set_title('Exp LSTA', fontsize=12, fontweight='bold')
                col += 1

            # Column 3 (or 2): Model LSTA
            ax_model = axes[img_idx, col]
            lsta_m = lsta_model[i, img_idx].copy()

            # Contrast enhancement
            lsta_m = np.sign(lsta_m) * (np.abs(lsta_m) ** expon_treat)
            vmax_m = np.max(np.abs(lsta_m)) * vmax_thresh
            if vmax_m > 0:
                im_model = ax_model.imshow(lsta_m, cmap=cmap, vmin=-vmax_m, vmax=vmax_m)
                plt.colorbar(im_model, ax=ax_model, fraction=0.046)
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

            ax_model.axis('off')
            if img_idx == 0:
                ax_model.set_title('Model LSTA', fontsize=12, fontweight='bold')

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