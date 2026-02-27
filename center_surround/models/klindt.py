import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Optional, Any

from .activations import build_activation_layer
from .regularizers import L1Smooth2DRegularizer


class KlindtCore2D(nn.Module):
    """
    Multi-layer 2D convolutional core with batch normalization and dropout.
    """
    
    def __init__(
        self,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        smoothness_reg: float,
        sparsity_reg: float,
        center_mass_reg: float | Iterable[float],
        init_scales: np.ndarray,
        init_kernels: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Parse kernel sizes
        self.kernel_sizes = []
        for k in kernel_sizes:
            if isinstance(k, int):
                self.kernel_sizes.append((k, k))
            elif isinstance(k, Iterable) and len(k) == 2:
                self.kernel_sizes.append(tuple(k))
            else:
                raise ValueError(f"Invalid kernel size format: {k}")
        
        self.num_kernels = list(num_kernels)
        self.act_fns = list(act_fns)
        self.kernel_constraint = kernel_constraint
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        input_channels = image_channels
        for i, (k_size, k_num, act_fn) in enumerate(zip(self.kernel_sizes, num_kernels, act_fns)):
            conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=k_num,
                kernel_size=k_size,
                stride=1,
                padding=0,
            )
            
            # Weight initialization
            if init_kernels is not None and init_kernels.startswith("gaussian"):
                with torch.no_grad():
                    weight = conv.weight
                    _, _, H, W = weight.shape
                    
                    # Parse sigma parameter
                    try:
                        sigma_str = init_kernels.split(":")[1]
                        sigma = float(sigma_str)
                    except (IndexError, ValueError):
                        sigma = 0.2
     
                    # Create 2D Gaussian template
                    yy, xx = torch.meshgrid(
                        torch.linspace(-1, 1, H),
                        torch.linspace(-1, 1, W),
                        indexing='ij'
                    )
                    gaussian = torch.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
                    gaussian = gaussian / gaussian.max()

                    # Initialize: Gaussian template × N(0, std)
                    init_noise = torch.randn_like(weight)
                    weight.copy_(init_noise * gaussian * init_scales[0][1])
            else:
                # Simple default: small random weights
                nn.init.normal_(conv.weight, mean=init_scales[0][0], std=init_scales[0][1])
            
            # Apply kernel constraint at init
            if kernel_constraint == 'norm':
                with torch.no_grad():
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2, 3), keepdim=True) + 1e-5)
                    conv.weight.data = conv.weight.data / norm
            
            self.conv_layers.append(conv)
            
            if self.bn_layers is not None:
                self.bn_layers.append(
                    nn.BatchNorm2d(
                        num_features=k_num,
                        affine=bn_cent,
                        track_running_stats=True,
                        momentum=0.02
                    )
                )
            
            input_channels = k_num
            self.activation_layers.append(build_activation_layer(act_fn))
        
        self.regularizer_module = L1Smooth2DRegularizer(
            sparsity_factor=sparsity_reg,
            smoothness_factor=smoothness_reg,
            center_mass_factor=center_mass_reg,
        )

    
    def apply_constraints(self):
        if self.kernel_constraint == 'norm':
            with torch.no_grad():
                for conv in self.conv_layers:
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2, 3), keepdim=True) + 1e-5)
                    conv.weight.data = conv.weight.data / norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.apply_constraints()
        x = self.dropout(x)
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.activation_layers[i](x)
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)
        
        return x
    
    def regularizer(self) -> torch.Tensor:
        return self.regularizer_module(self.conv_layers[-1].weight)


class KlindtReadout2D(nn.Module):
    """
    Spatial mask readout with feature weighting.
    """
    
    def __init__(
        self,
        num_kernels: Iterable[int],
        num_neurons: int,
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.reg = [mask_reg, weights_reg]
        self.final_relu = final_relu
        self.weights_constraint = weights_constraint
        self.mask_constraint = mask_constraint
        
        # Parse mask size
        if isinstance(mask_size, int):
            assert mask_size > 0
            self.mask_size = (mask_size, mask_size)
            self.num_mask_pixels = mask_size ** 2
        else:
            h, w = mask_size
            assert h > 0 and w > 0
            self.mask_size = (h, w)
            self.num_mask_pixels = h * w
        
        # Initialize mask weights
        if init_mask is not None:
            h, w = self.mask_size
            H, W = init_mask.shape[2], init_mask.shape[3]
            h_offset = (H - h) // 2
            w_offset = (W - w) // 2
            
            cropped = init_mask[:, :, h_offset:h_offset + h, w_offset:w_offset + w]
            reshaped = cropped.reshape(num_neurons, -1).T
            self.mask_weights = nn.Parameter(torch.tensor(reshaped, dtype=torch.float32))
        else:
            assert init_scales is not None, "Either init_mask or init_scales must be provided"
            mean, std = init_scales[1]
            mask_init = torch.normal(mean=mean, std=std, size=(self.num_mask_pixels, num_neurons))
            self.mask_weights = nn.Parameter(mask_init)
        
        # Initialize readout weights
        if init_weights is not None:
            self.readout_weights = nn.Parameter(init_weights)
        else:
            assert init_scales is not None, "Either init_weights or init_scales must be provided"
            mean, std = init_scales[2]
            num_kernels_list = list(num_kernels)
            self.readout_weights = nn.Parameter(
                torch.normal(mean=mean, std=std, size=(num_kernels_list[-1], num_neurons))
            )
        
        # Bias for final nonlinearity
        self.bias = nn.Parameter(torch.full((num_neurons,), 0.5)) if final_relu else None
    
    def apply_constraints(self):
        if self.mask_constraint == 'abs':
            with torch.no_grad():
                self.mask_weights.data = torch.abs(self.mask_weights.data)
        
        if self.weights_constraint == 'abs':
            with torch.no_grad():
                self.readout_weights.data = torch.abs(self.readout_weights.data)
        elif self.weights_constraint == 'norm':
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm
        elif self.weights_constraint == 'absnorm':
            with torch.no_grad():
                self.readout_weights.data = torch.abs(self.readout_weights.data)
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm
    
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()
        
        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"
        
        # Flatten spatial dims
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        
        # Apply spatial mask: [B, C, H*W] @ [H*W, N] -> [B, C, N]
        masked = torch.matmul(x_flat, self.mask_weights)
        masked = masked.permute(0, 2, 1)  # [B, N, C]
        
        # Apply readout weights: [B, N, C] * [1, N, C] -> sum -> [B, N]
        output = (masked * self.readout_weights.T.unsqueeze(0)).sum(dim=2)
        
        if self.final_relu:
            output = F.softplus(output + self.bias)
        
        return output
    
    def regularizer(self) -> torch.Tensor:
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[0]
        weights_reg = torch.mean(torch.sum(torch.abs(self.readout_weights), dim=0)) * self.reg[1]
        return mask_reg + weights_reg


class KlindtCoreReadout2D(nn.Module):
    """
    Complete Klindt-style CNN model combining core and readout.
    """
    
    def __init__(
        self,
        # Core parameters
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        init_scales: Iterable[Iterable[float]],  # 3x2: [kernel, mask, weights] x [mean, std]
        smoothness_reg: float = 1e0,
        sparsity_reg: float = 1e-1,
        center_mass_reg: float | Iterable[float] = 0,
        init_kernels: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        dropout_rate: float = 0.2,
        seed: Optional[int] = None,
        # Readout parameters
        num_neurons: int = 1,
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        mask_reg: float = 1e-3,
        weights_reg: float = 1e-1,
    ):
        super().__init__()
        
        # Convert init_scales to numpy array
        init_scales = np.array(init_scales)
        
        # Build core
        self.core = KlindtCore2D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            num_kernels=num_kernels,
            act_fns=act_fns,
            smoothness_reg=smoothness_reg,
            sparsity_reg=sparsity_reg,
            center_mass_reg=center_mass_reg,
            init_scales=init_scales,
            init_kernels=init_kernels,
            kernel_constraint=kernel_constraint,
            batch_norm=batch_norm,
            bn_cent=bn_cent,
            dropout_rate=dropout_rate,
            seed=seed,
        )
        
        # Compute output size of core
        if isinstance(image_size, int):
            dummy_input = torch.zeros(1, image_channels, image_size, image_size)
        else:
            dummy_input = torch.zeros(1, image_channels, *image_size)
        
        with torch.no_grad():
            core_out = self.core(dummy_input)
        _, _, h, w = core_out.shape
        self.mask_size = (h, w)
        
        # Build readout
        self.readout = KlindtReadout2D(
            num_kernels=num_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_mask=init_mask,
            init_weights=init_weights,
            init_scales=init_scales,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x
    
    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()
