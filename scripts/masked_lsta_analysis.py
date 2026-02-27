"""
Masked LSTA Analysis Script

Load a trained model and visualize LSTA comparisons across multiple mask conditions.
Supports KlindtCoreReadoutONOFF2D and KlindtCoreReadoutDedicatedONOFFMixed2D models.

Usage:
    python scripts/masked_lsta_analysis.py \
        --model-dir outputs/exp_13_data_4/klindtONOFF/run_XXXXXXXX_XXXXXX \
        --model-type onoff \
        --zoom 2.5 --vmax-thresh 0.6 --expon-treat 3
"""

import argparse
import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from center_surround.models.klindtSurround import (
    KlindtCoreReadoutONOFF2D,
    KlindtCoreReadoutDedicatedONOFFMixed2D,
)
from center_surround.utils import (
    compute_lsta,
    compute_lsta_masked,
    create_batch_masks_for_conditions,
)


# ============================================================
# Data paths (adjust if your experiment layout differs)
# ============================================================
IMAGES_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/natural_images_alsoWith_masks.pkl'
LSTA_EXP_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/768_pert/lSTAs_data_total_lstas.pkl'
RF_FITS_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/Chirp_Checkerboard_Analysis_mix/sta_data_mix.pkl'
RF_FITS_MASKING_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis_online/Checkerboard_Analysis_rec_0/sta_data_3D_fitted.pkl'
CELL_BATCHES_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/stims/spacing_1.3_vicinity_0.2/cell_batches_info.pkl'
ACCEPTED_LSTAS_PATH = '/home/ridvan/Documents/exp/13_exp/20251208/Analysis/lSTAs/accepted_lstas_lookup.pkl'

MASK_LABELS = ['Original', 'Mask 1', 'Mask 2', 'Mask 3', 'Mask 4', 'Mask 5']
SPACING_VICINITY = (1.3, 0.2)


# ============================================================
# Helper functions
# ============================================================

def gaussian2D(shape, amp, x0, y0, sigma_x, sigma_y, angle):
    """Generate 2D Gaussian for RF ellipse overlay."""
    if sigma_x == 0:
        sigma_x = 0.001
    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]), int(shape[1]))
    x = np.linspace(0, shape[1], shape[1])
    y = np.linspace(0, shape[0], shape[0])
    X, Y = np.meshgrid(x, y)

    theta = math.pi * angle / 180
    a = (math.cos(theta)**2) / (2 * sigma_x**2) + (math.sin(theta)**2) / (2 * sigma_y**2)
    b = -(math.sin(2 * theta)) / (4 * sigma_x**2) + (math.sin(2 * theta)) / (4 * sigma_y**2)
    c = (math.sin(theta)**2) / (2 * sigma_x**2) + (math.cos(theta)**2) / (2 * sigma_y**2)

    return amp * np.exp(-(a * (X - x0)**2 + 2 * b * (X - x0) * (Y - y0) + c * (Y - y0)**2))


def overlay_rf_ellipse(ax, ellipse_coords, img_shape, rf_dim, scale_factor=1.0,
                       level_factor=0.35, line_width=2, color='yellow', alpha=0.8):
    """Overlay RF ellipse contour on an axis."""
    if ellipse_coords is None or ellipse_coords[0] == 0:
        return

    img_h, img_w = img_shape
    center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + img_w / 2
    center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + img_h / 2

    scaled_ellipse = [
        ellipse_coords[0],
        center_x, center_y,
        ellipse_coords[3] * scale_factor,
        ellipse_coords[4] * scale_factor,
        ellipse_coords[5],
    ]

    gaussian = gaussian2D([img_h, img_w], *scaled_ellipse)
    ax.contour(
        np.abs(gaussian),
        levels=[level_factor * np.max(np.abs(gaussian))],
        colors=color,
        linewidths=line_width,
        alpha=alpha,
    )


def apply_zoom(ax, ellipse_coords, img_size, rf_dim, scale_factor, zoom):
    """Apply zoom around RF center."""
    if ellipse_coords is None or zoom <= 1.0:
        return
    center_x = (ellipse_coords[1] - rf_dim / 2) * scale_factor + img_size / 2
    center_y = (ellipse_coords[2] - rf_dim / 2) * scale_factor + img_size / 2
    zoom_half = (img_size / 2) / zoom
    ax.set_xlim(center_x - zoom_half, center_x + zoom_half)
    ax.set_ylim(center_y + zoom_half, center_y - zoom_half)


def apply_batch_mask_to_lsta(lsta, batch, rfs, bg_value=0.0, scale=6):
    """Apply elliptical masks for a batch onto an LSTA."""
    H, W = lsta.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    id_to_ellipse = {cid: ell for cid, ell in rfs}

    for cid in batch:
        if cid not in id_to_ellipse:
            continue
        ellipse = id_to_ellipse[cid]
        _, x, y, sx, sy, theta = ellipse
        center = (int(round(x * scale)), int(round(y * scale)))
        axes = (int(round(sx * scale * 0.8)), int(round(sy * scale * 0.8)))
        angle = np.degrees(theta)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)

    masked_lsta = np.where(mask == 255, lsta, bg_value).astype(lsta.dtype)
    return masked_lsta


# ============================================================
# Batch mask visualisation (debug)
# ============================================================

def visualize_batch_masks(masked_cells, all_RFs, img_size=432, scale=6, output_dir=None):
    """Visualize each batch's elliptical masks on a gray image."""
    num_batches = len(masked_cells)
    fig, axes = plt.subplots(1, num_batches + 1, figsize=(4 * (num_batches + 1), 4))

    id_to_ellipse = {cid: ell for cid, ell in all_RFs}

    # All RFs panel
    ax = axes[0]
    all_mask = np.ones((img_size, img_size), dtype=np.uint8) * 128
    for cid, ellipse in all_RFs:
        _, x, y, sx, sy, theta = ellipse
        center = (int(round(x * scale)), int(round(y * scale)))
        axes_size = (int(round(sx * scale * 0.8)), int(round(sy * scale * 0.8)))
        angle = np.degrees(theta)
        cv2.ellipse(all_mask, center, axes_size, angle, 0, 360, 255, thickness=1)
    ax.imshow(all_mask, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'All RFs ({len(all_RFs)} cells)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    for batch_idx, batch in enumerate(masked_cells):
        ax = axes[batch_idx + 1]
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cells_found = 0
        cells_missing = []

        for cid in batch:
            if cid not in id_to_ellipse:
                cells_missing.append(cid)
                continue
            cells_found += 1
            ellipse = id_to_ellipse[cid]
            _, x, y, sx, sy, theta = ellipse
            center = (int(round(x * scale)), int(round(y * scale)))
            axes_size = (int(round(sx * scale * 0.8)), int(round(sy * scale * 0.8)))
            angle = np.degrees(theta)
            cv2.ellipse(mask, center, axes_size, angle, 0, 360, 255, thickness=-1)

        masked_img = np.where(mask == 255, 255, 128).astype(np.uint8)
        ax.imshow(masked_img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Batch {batch_idx + 1}\n{cells_found}/{len(batch)} cells', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if cells_missing:
            print(f"Batch {batch_idx + 1}: Missing cells {cells_missing}")

    plt.suptitle('Batch Mask Verification (white = masked region)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, 'batch_masks_verification.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved batch mask verification to {save_path}")
    else:
        plt.show()

    print("\n=== Batch Masking Summary ===")
    print(f"Total RFs available: {len(all_RFs)}")
    for batch_idx, batch in enumerate(masked_cells):
        found = sum(1 for cid in batch if cid in id_to_ellipse)
        print(f"Batch {batch_idx + 1}: {found}/{len(batch)} cells found")


# ============================================================
# Main LSTA plotting function
# ============================================================

def plot_lsta_all_masks(
    neuron_idx,
    images_all,
    lsta_model_all,
    lsta_exp_all,
    mask_labels,
    rf_fits=None,
    cell_ids=None,
    rf_dim=72,
    cmap='RdBu_r',
    vmax_thresh=0.5,
    expon_treat=3,
    zoom=1.0,
    target_size=432,
    output_dir=None,
    accepted_lstas=None,
    masked_cells=None,
    all_RFs=None,
    smooth_sigma=5,
    hide_non_accepted=True,
):
    """
    Plot LSTA comparison for one neuron with all mask variations in rows.
    Each row: one mask condition.
    Each column group: [Image, Exp LSTA, Model LSTA] for each of the 4 images.
    """
    num_masks = len(images_all)
    num_images = images_all[0].shape[0]
    panels_per_image = 3  # Image, Exp, Model

    # Get cell ID and RF ellipse
    cell_id = None
    ellipse_coords = None
    if cell_ids is not None and neuron_idx < len(cell_ids):
        cell_id = cell_ids[neuron_idx]
        if rf_fits is not None and cell_id in rf_fits:
            ellipse_coords = rf_fits[cell_id]['center_analyse']['EllipseCoor']

    # Figure setup
    panel_width = 2.0
    panel_height = 2.0
    group_spacing = 0.4

    fig_width = num_images * panels_per_image * panel_width + (num_images - 1) * group_spacing + 1.5
    fig_height = num_masks * panel_height + 1.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    width_ratios = [0.3]  # left margin for row labels
    for img_idx in range(num_images):
        for _ in range(panels_per_image):
            width_ratios.append(1)
        if img_idx < num_images - 1:
            width_ratios.append(0.3)  # spacer

    total_cols = len(width_ratios)
    gs = GridSpec(num_masks, total_cols, figure=fig, width_ratios=width_ratios,
                  wspace=0.05, hspace=0.15)

    scale_factor = target_size / rf_dim

    for mask_idx in range(num_masks):
        images = images_all[mask_idx]
        lsta_model = lsta_model_all[mask_idx]
        lsta_exp = lsta_exp_all[mask_idx]

        # Row label
        ax_label = fig.add_subplot(gs[mask_idx, 0])
        ax_label.text(0.5, 0.5, mask_labels[mask_idx], rotation=90,
                      va='center', ha='center', fontsize=10, fontweight='bold')
        ax_label.axis('off')

        for img_idx in range(num_images):
            base_col = 1 + img_idx * panels_per_image + img_idx

            # Check acceptance
            is_accepted = False
            if accepted_lstas is not None and cell_id is not None:
                if cell_id in accepted_lstas:
                    if mask_idx in accepted_lstas[cell_id]:
                        is_accepted = accepted_lstas[cell_id][mask_idx].get(img_idx + 1, False)
            is_original = (mask_idx == 0)
            show_lsta = (is_accepted or is_original) if hide_non_accepted else True

            # --- Original Image ---
            ax_img = fig.add_subplot(gs[mask_idx, base_col])
            img_data = images[img_idx, 0]
            if img_data.shape[0] != target_size:
                img_display = cv2.resize(img_data, (target_size, target_size), interpolation=cv2.INTER_AREA)
            else:
                img_display = img_data
            ax_img.imshow(img_display, cmap='gray')
            if ellipse_coords is not None:
                overlay_rf_ellipse(ax_img, ellipse_coords, (target_size, target_size),
                                   rf_dim, scale_factor=scale_factor, color='red')
                apply_zoom(ax_img, ellipse_coords, target_size, rf_dim, scale_factor, zoom)
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if mask_idx == 0:
                ax_img.set_title(f'Img {img_idx + 1}', fontsize=9)

            # --- Experimental LSTA ---
            ax_exp = fig.add_subplot(gs[mask_idx, base_col + 1])
            if show_lsta:
                lsta_e = lsta_exp[neuron_idx, img_idx].copy()
                if mask_idx > 0 and masked_cells is not None and all_RFs is not None:
                    batch = masked_cells[mask_idx - 1]
                    lsta_e = apply_batch_mask_to_lsta(lsta_e, batch, all_RFs, bg_value=0.0, scale=6)
                if target_size != 432:
                    lsta_e = cv2.resize(lsta_e, (target_size, target_size), interpolation=cv2.INTER_AREA)
                lsta_e = np.sign(lsta_e) * (np.abs(lsta_e) ** expon_treat)
                if smooth_sigma and smooth_sigma > 0:
                    lsta_e = gaussian_filter(lsta_e, sigma=smooth_sigma)
                vmax_e = np.max(np.abs(lsta_e)) * vmax_thresh
                if vmax_e > 0:
                    ax_exp.imshow(lsta_e, cmap=cmap, vmin=-vmax_e, vmax=vmax_e)
                else:
                    ax_exp.imshow(lsta_e, cmap=cmap)
                if ellipse_coords is not None:
                    overlay_rf_ellipse(ax_exp, ellipse_coords, (target_size, target_size),
                                       rf_dim, scale_factor=scale_factor, color='red')
                    apply_zoom(ax_exp, ellipse_coords, target_size, rf_dim, scale_factor, zoom)
                border_color = 'green' if is_accepted else 'black'
                border_width = 3.0 if is_accepted else 1.5
            else:
                ax_exp.imshow(np.ones((target_size, target_size)), cmap='gray', vmin=0, vmax=1)
                border_color = 'lightgray'
                border_width = 1.5
            for spine in ax_exp.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_width)
                spine.set_color(border_color)
            ax_exp.set_xticks([])
            ax_exp.set_yticks([])
            if mask_idx == 0:
                ax_exp.set_title(f'Exp {img_idx + 1}', fontsize=9)

            # --- Model LSTA ---
            ax_model = fig.add_subplot(gs[mask_idx, base_col + 2])
            if show_lsta:
                lsta_m = lsta_model[neuron_idx, img_idx].copy()
                if lsta_m.shape[0] != target_size:
                    lsta_m = cv2.resize(lsta_m, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                lsta_m = np.sign(lsta_m) * (np.abs(lsta_m) ** expon_treat)
                if smooth_sigma and smooth_sigma > 0:
                    lsta_m = gaussian_filter(lsta_m, sigma=smooth_sigma)
                vmax_m = np.max(np.abs(lsta_m)) * vmax_thresh
                if vmax_m > 0:
                    ax_model.imshow(lsta_m, cmap=cmap, vmin=-vmax_m, vmax=vmax_m)
                else:
                    ax_model.imshow(lsta_m, cmap=cmap)
                if ellipse_coords is not None:
                    overlay_rf_ellipse(ax_model, ellipse_coords, (target_size, target_size),
                                       rf_dim, scale_factor=scale_factor, color='red')
                    apply_zoom(ax_model, ellipse_coords, target_size, rf_dim, scale_factor, zoom)
                model_border_color = 'green' if is_accepted else 'black'
                model_border_width = 3.0 if is_accepted else 1.5
            else:
                ax_model.imshow(np.ones((target_size, target_size)), cmap='gray', vmin=0, vmax=1)
                model_border_color = 'lightgray'
                model_border_width = 1.5
            for spine in ax_model.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(model_border_width)
                spine.set_color(model_border_color)
            ax_model.set_xticks([])
            ax_model.set_yticks([])
            if mask_idx == 0:
                ax_model.set_title(f'Model {img_idx + 1}', fontsize=9)

    title = f'Neuron {neuron_idx}'
    if cell_id is not None:
        title += f' (Cell {cell_id})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(os.path.join(output_dir, 'lsta_all_masks'), exist_ok=True)
        filename = f'neuron_{neuron_idx:03d}'
        if cell_id is not None:
            filename += f'_cell_{cell_id}'
        filename += '_lsta_all_masks.png'
        save_path = os.path.join(output_dir, 'lsta_all_masks', filename)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


# ============================================================
# Model loading
# ============================================================

MODEL_CLASSES = {
    'onoff': KlindtCoreReadoutONOFF2D,
    'dedicated': KlindtCoreReadoutDedicatedONOFFMixed2D,
}

MODEL_WEIGHT_NAMES = {
    'onoff': 'best_model_onoff.pth',
    'dedicated': 'best_model_dedicated_onoff_mixed.pth',
}


def load_model(model_dir, model_type, device):
    """Load model config and weights from a run directory."""
    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    model_config = config['model_config']
    picked_cells = config['picked_cells']

    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print(f"\nNumber of cells: {len(picked_cells)}")

    ModelClass = MODEL_CLASSES[model_type]
    model = ModelClass(**model_config)

    # Try to find the model weights file
    weight_name = MODEL_WEIGHT_NAMES[model_type]
    model_path = os.path.join(model_dir, weight_name)
    if not os.path.exists(model_path):
        # Fallback: look for any .pth file
        pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if len(pth_files) == 1:
            model_path = os.path.join(model_dir, pth_files[0])
            print(f"Using fallback weight file: {pth_files[0]}")
        elif pth_files:
            # Prefer files with 'best' in the name
            best_files = [f for f in pth_files if 'best' in f.lower()]
            model_path = os.path.join(model_dir, best_files[0] if best_files else pth_files[0])
            print(f"Using weight file: {os.path.basename(model_path)}")
        else:
            raise FileNotFoundError(f"No .pth files found in {model_dir}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    return model, model_config, picked_cells


# ============================================================
# Data loading
# ============================================================

def load_images(input_size):
    """Load and preprocess natural images for all mask conditions."""
    with open(IMAGES_PATH, 'rb') as f:
        all_images = pickle.load(f)

    print(f"Loaded images shape: {all_images.shape}")
    num_images = all_images.shape[0]
    num_masks = all_images.shape[1]
    print(f"Number of images: {num_images}, Number of mask variations: {num_masks}")

    lsta_images_all = {}
    for mask_idx in range(num_masks):
        imgs = all_images[:, mask_idx, :, :]
        imgs = (imgs - imgs.mean(axis=(1, 2), keepdims=True)) * 5
        imgs_resized = np.zeros((imgs.shape[0], input_size, input_size), dtype=imgs.dtype)
        for idx in range(imgs.shape[0]):
            imgs_resized[idx] = cv2.resize(imgs[idx], (input_size, input_size), interpolation=cv2.INTER_AREA)
        lsta_images_all[mask_idx] = imgs_resized[:, np.newaxis, :, :]

    print(f"Preprocessed images for {num_masks} mask conditions, each shape: {lsta_images_all[0].shape}")
    return lsta_images_all, num_masks


def load_experimental_data(picked_cells, num_masks):
    """Load RF fits, experimental LSTA, cell batches, and accepted LSTA lookup."""
    # RF fits for plotting
    with open(RF_FITS_PATH, 'rb') as f:
        rf_fits = pickle.load(f)

    # Experimental LSTA (72x72 -> upsampled to 432x432)
    with open(LSTA_EXP_PATH, 'rb') as f:
        all_lsta_exp_raw = pickle.load(f)

    lsta_exp_all = {}
    for mask_idx in range(num_masks):
        lsta_exp = []
        for cell_id in picked_cells:
            cell_lstas = all_lsta_exp_raw[mask_idx][cell_id]  # (4, 72, 72)
            upsampled = np.repeat(np.repeat(cell_lstas, 6, axis=1), 6, axis=2)
            lsta_exp.append(upsampled)
        lsta_exp_all[mask_idx] = np.array(lsta_exp)

    print(f"Loaded experimental LSTA for {num_masks} conditions, shape: {lsta_exp_all[0].shape}")

    # Cell batches for masking
    with open(CELL_BATCHES_PATH, 'rb') as f:
        cell_batches = pickle.load(f)

    masks_batches = cell_batches[SPACING_VICINITY]
    masked_cells = []
    for cell_masks in masks_batches:
        masked_cells.append([cid for cid in cell_masks['selected']])

    print(f"Loaded {len(masked_cells)} mask batches")
    for i, batch in enumerate(masked_cells):
        print(f"  Batch {i + 1}: {len(batch)} cells")

    # RF fits used for creating mask batches (online analysis)
    with open(RF_FITS_MASKING_PATH, 'rb') as f:
        rf_fits_masking = pickle.load(f)

    all_RFs = []
    for key, item in rf_fits_masking.items():
        ellipse = item['center_analyse']['EllipseCoor']
        all_RFs.append((key, ellipse))

    print(f"RF ellipses for masking: {len(all_RFs)} cells")
    print(f"RF ellipses for plotting: {len(rf_fits)} cells")

    # Accepted LSTA lookup
    with open(ACCEPTED_LSTAS_PATH, 'rb') as f:
        accepted_lstas = pickle.load(f)
    print(f"Loaded accepted_lstas for {len(accepted_lstas)} cells")

    return rf_fits, lsta_exp_all, masked_cells, all_RFs, accepted_lstas


# ============================================================
# LSTA computation
# ============================================================

def compute_all_lsta(model, lsta_images_all, rf_masks_by_condition, num_masks, device,
                     use_masked=True):
    """Compute model LSTA for all mask conditions."""
    lsta_model_all = {}
    for mask_idx in range(num_masks):
        if use_masked:
            rf_masks = rf_masks_by_condition[mask_idx]
            lsta_model_all[mask_idx] = compute_lsta_masked(
                model, lsta_images_all[mask_idx], rf_masks, device=device
            )
            print(f"Mask {mask_idx} ({MASK_LABELS[mask_idx]}): Masked LSTA shape = {lsta_model_all[mask_idx].shape}")
        else:
            lsta_model_all[mask_idx] = compute_lsta(model, lsta_images_all[mask_idx], device=device)
            print(f"Mask {mask_idx} ({MASK_LABELS[mask_idx]}): LSTA shape = {lsta_model_all[mask_idx].shape}")

    mode = 'MASKED (batch-based)' if use_masked else 'STANDARD'
    print(f"\nUsing {mode} LSTA computation")
    return lsta_model_all


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Masked LSTA Analysis')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the trained model run directory')
    parser.add_argument('--model-type', type=str, default='onoff',
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type: onoff or dedicated (default: onoff)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: alongside model-dir)')
    parser.add_argument('--zoom', type=float, default=2.5,
                        help='Zoom factor around RF center (default: 2.5)')
    parser.add_argument('--vmax-thresh', type=float, default=0.6,
                        help='Fraction of max abs value for color scale (default: 0.6)')
    parser.add_argument('--expon-treat', type=float, default=3,
                        help='Contrast enhancement exponent (default: 3)')
    parser.add_argument('--smooth-sigma', type=float, default=5,
                        help='Gaussian smoothing sigma (default: 5, 0 to disable)')
    parser.add_argument('--target-size', type=int, default=432,
                        help='Display size for images/LSTAs (default: 432)')
    parser.add_argument('--no-masked-lsta', action='store_true',
                        help='Use standard (unmasked) LSTA instead of masked')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all LSTAs regardless of acceptance status')
    parser.add_argument('--single-neuron', type=int, default=None,
                        help='Plot only this neuron index (interactive)')
    parser.add_argument('--debug-masks', action='store_true',
                        help='Visualize batch masks for debugging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.model_dir),
            'lsta_analysis',
            os.path.basename(args.model_dir),
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load model
    model, model_config, picked_cells = load_model(args.model_dir, args.model_type, device)
    input_size = model_config['image_size']
    num_neurons = len(picked_cells)

    # Load images
    lsta_images_all, num_masks = load_images(input_size)

    # Load experimental data
    rf_fits, lsta_exp_all, masked_cells, all_RFs, accepted_lstas = \
        load_experimental_data(picked_cells, num_masks)

    # Debug: visualize batch masks
    if args.debug_masks:
        visualize_batch_masks(masked_cells, all_RFs, img_size=432, scale=6,
                              output_dir=args.output_dir)

    # Create RF masks for masked LSTA computation
    rf_masks_by_condition = create_batch_masks_for_conditions(
        masked_cells=masked_cells,
        all_RFs=all_RFs,
        num_neurons=num_neurons,
        mask_size=(input_size, input_size),
        rf_dim=72,
        ellipse_scale=0.8,
    )

    print(f"\nCreated RF masks for {len(rf_masks_by_condition)} conditions")
    for mask_idx, masks in rf_masks_by_condition.items():
        coverage = masks[0].sum() / (input_size * input_size)
        print(f"  Condition {mask_idx}: shape {masks.shape}, coverage {coverage:.2%}")

    # Compute model LSTA
    use_masked = not args.no_masked_lsta
    lsta_model_all = compute_all_lsta(
        model, lsta_images_all, rf_masks_by_condition, num_masks, device,
        use_masked=use_masked,
    )

    # Plot
    plot_kwargs = dict(
        images_all=lsta_images_all,
        lsta_model_all=lsta_model_all,
        lsta_exp_all=lsta_exp_all,
        mask_labels=MASK_LABELS,
        rf_fits=rf_fits,
        cell_ids=picked_cells,
        rf_dim=72,
        cmap='RdBu_r',
        vmax_thresh=args.vmax_thresh,
        expon_treat=args.expon_treat,
        zoom=args.zoom,
        target_size=args.target_size,
        accepted_lstas=accepted_lstas,
        masked_cells=masked_cells,
        all_RFs=all_RFs,
        smooth_sigma=args.smooth_sigma,
        hide_non_accepted=not args.show_all,
    )

    if args.single_neuron is not None:
        print(f"\nPlotting single neuron {args.single_neuron}...")
        plot_lsta_all_masks(
            neuron_idx=args.single_neuron,
            output_dir=None,  # show interactively
            **plot_kwargs,
        )
    else:
        print(f"\nPlotting all {num_neurons} neurons...")
        saved_paths = []
        for neuron_idx in range(num_neurons):
            path = plot_lsta_all_masks(
                neuron_idx=neuron_idx,
                output_dir=args.output_dir,
                **plot_kwargs,
            )
            saved_paths.append(path)
            if (neuron_idx + 1) % 10 == 0:
                print(f"  Processed {neuron_idx + 1}/{num_neurons} neurons")

        print(f"\nSaved {len(saved_paths)} plots to {args.output_dir}/lsta_all_masks/")


if __name__ == '__main__':
    main()
