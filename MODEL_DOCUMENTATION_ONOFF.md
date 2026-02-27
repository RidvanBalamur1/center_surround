## KlindtCoreReadoutONOFF2D - ON/OFF Polarity Model

**File:** `center_surround/models/klindtSurround.py`

### Overview

An extension of the basic Klindt model with **explicit ON/OFF kernel polarity**. This architecture enforces that:

- **ON kernels** have positive weights (detect light increments)
- **OFF kernels** have negative weights (detect light decrements)

This biologically-inspired constraint matches the known dichotomy of ON and OFF retinal ganglion cells.

### Architecture

```
Input Image [B, 1, H, W]
        │
        ▼
┌─────────────────────────┐
│   KlindtCoreONOFF2D     │
│  ┌───────────────────┐  │
│  │     Dropout       │  │
│  │         ▼         │  │
│  │ ┌───────┬───────┐ │  │
│  │ │  ON   │  OFF  │ │  │  ON kernels: weights ≥ 0
│  │ │kernels│kernels│ │  │  OFF kernels: weights ≤ 0
│  │ └───────┴───────┘ │  │
│  │         ▼         │  │
│  │    Activation     │  │  (ReLU)
│  │         ▼         │  │
│  │    BatchNorm      │  │
│  └───────────────────┘  │
└─────────────────────────┘
        │
        ▼ [B, n_on + n_off, h, w]
┌─────────────────────────┐
│   KlindtReadoutONOFF2D  │
│  ┌───────────────────┐  │
│  │   ON Features     │──►  ON Mask ──► ON Weights ──┐
│  │  [0:n_on]         │                              │
│  │                   │                              ▼
│  │                   │                           [B, N]
│  │  OFF Features     │──►  OFF Mask ──► OFF Weights─┘
│  │  [n_on:]          │
│  └───────────────────┘  │
└─────────────────────────┘
        │
        ▼
Output [B, num_neurons]
```

### Components

#### KlindtCoreONOFF2D

Convolutional core with polarity-constrained kernels.

**Key Differences from KlindtCore2D:**

- Kernels are split into ON and OFF groups
- ON kernels are constrained to be positive: `weight = |weight|`
- OFF kernels are constrained to be negative: `weight = -|weight|`
- Constraints are applied at every forward pass

**Polarity Constraint (applied each forward pass):**

```python
def apply_constraints(self):
    # ON kernels: ensure positive weights
    conv.weight[:n_on_kernels].data = torch.abs(conv.weight[:n_on_kernels].data)
    # OFF kernels: ensure negative weights
    conv.weight[n_on_kernels:].data = -torch.abs(conv.weight[n_on_kernels:].data)
```

**Key Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `n_on_kernels` | int | Number of ON (positive) kernels |
| `n_off_kernels` | int | Number of OFF (negative) kernels |
| `kernel_sizes` | list | Kernel size(s), e.g., `[24, 24]` |

#### KlindtReadoutONOFF2D

Readout with separate spatial masks for ON and OFF pathways.

**Key Differences from KlindtReadout2D:**

- Two spatial masks per neuron: ON mask and OFF mask
- ON mask pools only from ON kernel features
- OFF mask pools only from OFF kernel features
- Separate channel weights for each pathway

**Forward Pass:**

```python
# Split features by pathway
on_features = x[:, :n_on_kernels, :, :]   # [B, n_on, H, W]
off_features = x[:, n_on_kernels:, :, :]  # [B, n_off, H, W]

# ON pathway: ON mask × ON features × ON weights
on_output = (on_masked * on_weights).sum(dim=1)

# OFF pathway: OFF mask × OFF features × OFF weights
off_output = (off_masked * off_weights).sum(dim=1)

# Combine
output = on_output + off_output
```

**Learned Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `mask_weights` | `[H*W, 2, num_neurons]` | 2 masks: index 0 = ON, index 1 = OFF |
| `on_weights` | `[n_on_kernels, num_neurons]` | Weights for ON channels |
| `off_weights` | `[n_off_kernels, num_neurons]` | Weights for OFF channels |

### Usage Example

```python
from center_surround.models.klindtSurround import KlindtCoreReadoutONOFF2D

model = KlindtCoreReadoutONOFF2D(
    # Core parameters
    image_size=108,
    image_channels=1,
    kernel_sizes=[24, 24],  # Two values for ON and OFF kernel sizes
    n_on_kernels=2,         # Number of ON kernels
    n_off_kernels=2,        # Number of OFF kernels
    act_fns=['relu'],
    init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    init_kernels='gaussian:0.28',
    smoothness_reg=1e-4,
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

# Training
loss = mse_loss(outputs, targets) + model.regularizer()
```

### Visualizing ON/OFF Masks

The ON/OFF model allows direct visualization of each cell's polarity preference:

```python
# Get mask weights: [H*W, 2, num_neurons]
mask_weights = model.readout.mask_weights.detach().cpu().numpy()

# Reshape to spatial
H, W = model.mask_size
on_masks = mask_weights[:, 0, :].reshape(H, W, -1)   # ON masks
off_masks = mask_weights[:, 1, :].reshape(H, W, -1)  # OFF masks

# For neuron i:
# - Strong ON mask + weak OFF mask → ON cell
# - Weak ON mask + strong OFF mask → OFF cell
# - Both strong but spatially offset → ON-OFF or center-surround cell
```

### Interpretation

**Cell Type Classification:**
| ON Mask | OFF Mask | Cell Type |
|---------|----------|-----------|
| Strong | Weak | ON cell |
| Weak | Strong | OFF cell |
| Strong | Strong (same location) | ON-OFF cell |
| Strong (center) | Strong (surround) | Center-surround |

**Kernel Interpretation:**

- ON kernels respond to bright spots/edges on dark background
- OFF kernels respond to dark spots/edges on bright background
- The ReLU activation after convolution means:
  - ON kernel + bright stimulus → positive activation
  - OFF kernel + dark stimulus → positive activation (since -weight × -input = +)

---

## Model Comparison

| Feature                    | KlindtCoreReadout2D | KlindtCoreReadoutONOFF2D       |
| -------------------------- | ------------------- | ------------------------------ |
| Kernel polarity            | Unconstrained       | ON (positive) / OFF (negative) |
| Number of masks per neuron | 1                   | 2 (ON + OFF)                   |
| Pathway separation         | No                  | Yes                            |
| Interpretability           | Lower               | Higher                         |
| Parameters                 | Fewer               | More                           |
| Best for                   | General prediction  | Polarity analysis              |

---
