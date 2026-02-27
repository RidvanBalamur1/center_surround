# Model Documentation

This document describes the CNN models for Retinal Ganglion Cell (RGC) response prediction.

---

## Table of Contents

1. [KlindtCoreReadout2D - Basic Model](#klindtcorereadout2d---basic-model)
2. [KlindtCoreReadoutONOFF2D - ON/OFF Polarity Model](#klindtcorereadoutonoff2d---onoff-polarity-model)

---

## KlindtCoreReadout2D - Basic Model

**File:** `center_surround/models/klindt.py`

### Overview

The basic Klindt-style CNN model for predicting RGC responses to visual stimuli. The architecture follows the approach from Klindt et al., combining a convolutional feature extractor (core) with a spatial readout mechanism.

### Architecture

```
Input Image [B, 1, H, W]
        │
        ▼
┌───────────────────┐
│   KlindtCore2D    │  Convolutional feature extraction
│  ┌─────────────┐  │
│  │   Dropout   │  │
│  │      ▼      │  │
│  │  Conv2D     │  │  kernel_size × kernel_size
│  │      ▼      │  │
│  │ Activation  │  │  (ReLU, ELU, etc.)
│  │      ▼      │  │
│  │ BatchNorm   │  │  (optional)
│  └─────────────┘  │
└───────────────────┘
        │
        ▼ [B, num_kernels, h, w]
┌───────────────────┐
│  KlindtReadout2D  │  Spatial mask readout
│  ┌─────────────┐  │
│  │ Spatial     │  │  Weighted sum over spatial locations
│  │ Pooling     │  │  mask_weights: [h*w, num_neurons]
│  │      ▼      │  │
│  │ Channel     │  │  Weighted sum over feature channels
│  │ Weighting   │  │  readout_weights: [num_kernels, num_neurons]
│  │      ▼      │  │
│  │ (Softplus)  │  │  Optional output nonlinearity
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
Output [B, num_neurons]
```

### Components

#### KlindtCore2D

Multi-layer 2D convolutional core with batch normalization and dropout.

**Forward Pass:**

1. Apply dropout to input
2. For each layer:
   - Apply weight constraints (norm constraint)
   - Convolution (no padding)
   - Activation function
   - Batch normalization (if enabled)

**Key Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image_channels` | int | Number of input channels (1 for grayscale) |
| `kernel_sizes` | list | Kernel size(s) for conv layers, e.g., `[24]` |
| `num_kernels` | list | Number of filters per layer, e.g., `[4]` |
| `act_fns` | list | Activation function names, e.g., `['relu']` |
| `batch_norm` | bool | Whether to use batch normalization |
| `dropout_rate` | float | Dropout probability (applied to input) |
| `kernel_constraint` | str | `'norm'` to normalize kernel weights |

#### KlindtReadout2D

Spatial mask readout that learns a weighted combination of features across space and channels.

**Forward Pass:**

1. Flatten spatial dimensions: `[B, C, H, W] → [B, C, H*W]`
2. Apply spatial mask: `[B, C, H*W] @ [H*W, N] → [B, C, N]`
3. Apply channel weights: `[B, C, N] * [C, N] → sum → [B, N]`
4. Optional softplus nonlinearity

**Key Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `num_neurons` | int | Number of output neurons |
| `mask_size` | int/tuple | Spatial size of feature maps (auto-computed) |
| `final_relu` | bool | Apply softplus output nonlinearity |
| `mask_constraint` | str | `'abs'` for non-negative masks |
| `weights_constraint` | str | `'abs'`, `'norm'`, or `'absnorm'` |

### Regularization

The model uses several regularization terms:

1. **Smoothness Regularization** (`smoothness_reg`): Encourages smooth kernel weights using Laplacian penalty
2. **Sparsity Regularization** (`sparsity_reg`): L1 penalty on kernel weights
3. **Center Mass Regularization** (`center_mass_reg`): Encourages kernels to be centered
4. **Mask Regularization** (`mask_reg`): L1 penalty on spatial mask weights
5. **Weights Regularization** (`weights_reg`): L1 penalty on readout weights

**Total Loss:**

```python
loss = mse_loss + model.regularizer()
```

### Initialization

**Kernel Initialization:**

- `init_kernels='gaussian:0.28'`: Initialize with Gaussian-shaped weights
  - Creates a 2D Gaussian template centered on the kernel
  - Multiplies by random noise scaled by `init_scales[0][1]`
- Default: Normal distribution with `mean=init_scales[0][0]`, `std=init_scales[0][1]`

**Init Scales Format:**

```python
init_scales = [
    [mean, std],  # Kernel weights
    [mean, std],  # Mask weights
    [mean, std],  # Readout weights
]
# Example: [[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]]
```

### Usage Example

```python
from center_surround.models.klindt import KlindtCoreReadout2D

model = KlindtCoreReadout2D(
    # Core parameters
    image_size=108,
    image_channels=1,
    kernel_sizes=[24],
    num_kernels=[4],
    act_fns=['relu'],
    init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    init_kernels='gaussian:0.28',
    smoothness_reg=1e-4,
    sparsity_reg=0,
    dropout_rate=0.2,
    batch_norm=True,
    kernel_constraint='norm',

    # Readout parameters
    num_neurons=56,
    final_relu=True,
    mask_constraint='abs',
    weights_constraint='abs',
    mask_reg=1e-4,
    weights_reg=1e-4,
)

# Forward pass
outputs = model(images)  # [B, num_neurons]

# Training with regularization
loss = mse_loss(outputs, targets) + model.regularizer()
```

### Learned Parameters

| Component | Parameter               | Shape                        | Description                        |
| --------- | ----------------------- | ---------------------------- | ---------------------------------- |
| Core      | `conv_layers[i].weight` | `[out_ch, in_ch, kH, kW]`    | Convolutional kernels              |
| Core      | `conv_layers[i].bias`   | `[out_ch]`                   | Convolutional biases               |
| Core      | `bn_layers[i].*`        | varies                       | Batch norm parameters              |
| Readout   | `mask_weights`          | `[H*W, num_neurons]`         | Spatial attention masks            |
| Readout   | `readout_weights`       | `[num_kernels, num_neurons]` | Channel weights                    |
| Readout   | `bias`                  | `[num_neurons]`              | Output bias (if `final_relu=True`) |

---

## Other Model Variants

### KlindtCoreReadoutPerChannel2D

**File:** `center_surround/models/klindtSurround.py`

Per-channel spatial masks instead of shared mask. Each feature channel has its own spatial mask, allowing different features to be pooled from different locations.

### KlindtCoreReadoutNMasks2D

**File:** `center_surround/models/klindtSurround.py`

Configurable number of spatial masks (N masks) where each mask pools from ALL channels. Useful for explicit center-surround modeling with 2 masks.

---

## Training Tips

1. **Initialization**: Use `init_kernels='gaussian:0.28'` for better convergence
2. **Constraints**: Use `mask_constraint='abs'` to ensure non-negative spatial attention
3. **Regularization**: Start with small values (1e-4) and adjust based on overfitting
4. **Learning Rate**: Typical values: 1e-3 to 1e-4 with Adam optimizer
5. **Batch Size**: 32-128 depending on GPU memory
6. **Early Stopping**: Monitor validation correlation, not just loss
