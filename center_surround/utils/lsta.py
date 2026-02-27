
import torch
import numpy as np
import cv2
from typing import Optional


def compute_lsta(model, images: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """
    Compute Local Spike-Triggered Average (LSTA) using gradients.

    The LSTA is computed as the gradient of the model output with respect to
    the input images, which shows which parts of the image each neuron is
    sensitive to.

    Args:
        model: Trained model (KlindtCoreReadout2D)
        images: Input images tensor, shape [B, C, H, W]
        device: Device to run computation on

    Returns:
        lsta_array: shape [num_neurons, B, H, W] - gradient maps per neuron
    """
    model = model.to(device)
    model.eval()

    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images).float()

    images = images.to(device)
    images.requires_grad_(True)

    # Forward pass
    outputs = model(images)  # [B, num_neurons]
    num_neurons = outputs.shape[1]

    lsta_per_neuron = []

    for neuron_idx in range(num_neurons):
        # Create gradient mask for this neuron
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, neuron_idx] = 1.0

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=images,
            grad_outputs=grad_outputs,
            retain_graph=True,
            only_inputs=True,
        )[0]  # [B, C, H, W]

        lsta_per_neuron.append(grads.detach().cpu().numpy())

    # Stack: [num_neurons, B, C, H, W]
    lsta_array = np.stack(lsta_per_neuron, axis=0)

    # Squeeze channel dimension if single channel -> [num_neurons, B, H, W]
    if lsta_array.shape[2] == 1:
        lsta_array = lsta_array.squeeze(2)

    return lsta_array


def compute_lsta_masked(
    model,
    images: torch.Tensor,
    rf_masks: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Compute masked LSTA - gradients are zeroed outside RF region.

    This matches experimental LSTA methodology where perturbations
    only occur inside the RF mask region.

    Args:
        model: Trained model
        images: Input images tensor, shape [B, C, H, W]
        rf_masks: Binary RF masks, shape [num_neurons, H, W]
                  Values should be 1 inside RF, 0 outside.
                  H, W should match the LSTA output size (model input size).
        device: Device to run computation on

    Returns:
        masked_lsta: shape [num_neurons, B, H, W] - masked gradient maps
    """
    # Compute standard LSTA
    lsta = compute_lsta(model, images, device)  # [N, B, H, W]

    num_neurons, num_images, H, W = lsta.shape

    # Ensure rf_masks has correct shape
    assert rf_masks.shape[0] == num_neurons, \
        f"rf_masks has {rf_masks.shape[0]} neurons but LSTA has {num_neurons}"

    # Resize rf_masks if spatial dimensions don't match
    if rf_masks.shape[1] != H or rf_masks.shape[2] != W:
        rf_masks_resized = np.zeros((num_neurons, H, W), dtype=rf_masks.dtype)
        for i in range(num_neurons):
            rf_masks_resized[i] = cv2.resize(
                rf_masks[i].astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )
        rf_masks = rf_masks_resized

    # Apply mask per neuron (broadcast over batch dimension)
    # rf_masks: [N, H, W] -> [N, 1, H, W]
    masked_lsta = lsta * rf_masks[:, None, :, :]

    return masked_lsta


def _draw_ellipse_on_mask(
    mask: np.ndarray,
    ellipse_coords,
    rf_dim: int,
    scale: float,
    ellipse_scale: float,
    H: int,
    W: int
) -> bool:
    """
    Draw a single ellipse on the mask array.

    Args:
        mask: Array to draw on
        ellipse_coords: [amp, x, y, sigma_x, sigma_y, theta] or None
        rf_dim: Original RF dimension (72)
        scale: Scale factor from rf_dim to mask size
        ellipse_scale: Scale factor for ellipse axes
        H, W: Mask dimensions

    Returns:
        True if ellipse was drawn, False if invalid ellipse
    """
    if ellipse_coords is None or ellipse_coords[0] == 0:
        return False

    _, x, y, sigma_x, sigma_y, theta = ellipse_coords

    # Convert from RF coordinates to mask coordinates
    center_x = int(round((x - rf_dim / 2) * scale + W / 2))
    center_y = int(round((y - rf_dim / 2) * scale + H / 2))

    # Scale ellipse axes
    axes_x = int(round(sigma_x * scale * ellipse_scale))
    axes_y = int(round(sigma_y * scale * ellipse_scale))

    # Angle in degrees
    angle = np.degrees(theta)

    # Draw filled ellipse
    cv2.ellipse(
        mask,
        center=(center_x, center_y),
        axes=(axes_x, axes_y),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1  # Filled
    )
    return True


def create_rf_masks_from_ellipses(
    rf_fits: dict,
    cell_ids: list,
    mask_size: tuple,
    rf_dim: int = 72,
    ellipse_scale: float = 0.8,
) -> np.ndarray:
    """
    Create binary RF masks from ellipse fits (one per neuron).

    Args:
        rf_fits: Dictionary with RF ellipse parameters per cell.
                 Expected structure: rf_fits[cell_id]['center_analyse']['EllipseCoor']
                 where EllipseCoor = [amp, x, y, sigma_x, sigma_y, theta]
        cell_ids: List of cell IDs in neuron order
        mask_size: Output mask size (H, W)
        rf_dim: Original RF dimension (default 72)
        ellipse_scale: Scale factor for ellipse axes (default 0.8)

    Returns:
        rf_masks: Binary masks, shape [num_neurons, H, W]
    """
    H, W = mask_size
    num_neurons = len(cell_ids)
    rf_masks = np.zeros((num_neurons, H, W), dtype=np.float32)

    # Scale factor from rf_dim to mask_size
    scale = H / rf_dim

    for i, cell_id in enumerate(cell_ids):
        if cell_id not in rf_fits:
            rf_masks[i] = 1.0
            continue

        ellipse_coords = rf_fits[cell_id]['center_analyse']['EllipseCoor']
        drawn = _draw_ellipse_on_mask(
            rf_masks[i], ellipse_coords, rf_dim, scale, ellipse_scale, H, W
        )

        if not drawn:
            rf_masks[i] = 1.0

    return rf_masks


def create_batch_mask_from_ellipses(
    batch_cell_ids: list,
    all_RFs: list,
    mask_size: tuple,
    rf_dim: int = 72,
    ellipse_scale: float = 0.8,
) -> np.ndarray:
    """
    Create a single binary mask from a batch of cell RF ellipses.

    This is used for masked LSTA computation where the mask should match
    the experimental condition: all cells in a batch share the same mask region.

    Args:
        batch_cell_ids: List of cell IDs whose RFs define the mask
        all_RFs: List of (cell_id, ellipse_coords) tuples for all cells
        mask_size: Output mask size (H, W)
        rf_dim: Original RF dimension (default 72)
        ellipse_scale: Scale factor for ellipse axes (default 0.8)

    Returns:
        mask: Binary mask, shape [H, W], 1 inside combined RF, 0 outside
    """
    H, W = mask_size
    mask = np.zeros((H, W), dtype=np.float32)

    # Scale factor from rf_dim to mask_size
    scale = H / rf_dim

    # Build lookup from cell_id to ellipse coords
    id_to_ellipse = {cid: ell for cid, ell in all_RFs}

    for cell_id in batch_cell_ids:
        if cell_id not in id_to_ellipse:
            continue
        ellipse_coords = id_to_ellipse[cell_id]
        _draw_ellipse_on_mask(mask, ellipse_coords, rf_dim, scale, ellipse_scale, H, W)

    # If no ellipses were drawn, use entire image
    if mask.max() == 0:
        mask = np.ones((H, W), dtype=np.float32)

    return mask


def create_batch_masks_for_conditions(
    masked_cells: list,
    all_RFs: list,
    num_neurons: int,
    mask_size: tuple,
    rf_dim: int = 72,
    ellipse_scale: float = 0.8,
) -> dict:
    """
    Create RF masks for each mask condition (batch).

    In the experiment, each mask condition uses a different batch of cells,
    and ALL neurons share the same mask for a given condition.

    Args:
        masked_cells: List of batches, where masked_cells[i] is a list of cell IDs
                      for mask condition i+1 (mask_idx=0 is original, no masking)
        all_RFs: List of (cell_id, ellipse_coords) tuples for all cells
        num_neurons: Number of neurons being modeled
        mask_size: Output mask size (H, W)
        rf_dim: Original RF dimension (default 72)
        ellipse_scale: Scale factor for ellipse axes (default 0.8)

    Returns:
        masks_by_condition: dict mapping mask_idx -> array of shape [num_neurons, H, W]
                           For mask_idx=0 (original), returns all-ones masks
                           For mask_idx>0, all neurons share the same batch mask
    """
    H, W = mask_size
    masks_by_condition = {}

    # mask_idx=0: Original images, no masking needed (use full image)
    masks_by_condition[0] = np.ones((num_neurons, H, W), dtype=np.float32)

    # mask_idx=1,2,3,...: Each uses the corresponding batch
    for batch_idx, batch_cell_ids in enumerate(masked_cells):
        mask_idx = batch_idx + 1  # mask_idx=1 corresponds to batch 0

        # Create single batch mask
        batch_mask = create_batch_mask_from_ellipses(
            batch_cell_ids, all_RFs, mask_size, rf_dim, ellipse_scale
        )

        # Replicate for all neurons (they all share the same mask for this condition)
        masks_by_condition[mask_idx] = np.tile(batch_mask[None, :, :], (num_neurons, 1, 1))

    return masks_by_condition


def apply_batch_mask_to_lsta(lsta, batch, rfs, bg_value=0.0, scale=6):
    """
    Apply elliptical masks for a batch onto an LSTA.

    Masks the LSTA so only the regions covered by the batch's RF ellipses
    are visible. Used for masking experimental LSTA to match experimental
    conditions where only certain RF regions were stimulated.

    Args:
        lsta: 2D array [H, W] - LSTA data to mask
        batch: list of cell IDs whose RFs define the mask
        rfs: list of (cell_id, ellipse_coords) tuples
        bg_value: value for pixels outside the mask
        scale: scale factor from RF coordinates to LSTA coordinates

    Returns:
        masked_lsta: [H, W] with pixels outside RF regions set to bg_value
    """
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