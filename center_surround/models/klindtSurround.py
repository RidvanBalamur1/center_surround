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


class KlindtReadoutPerChannel2D(nn.Module):
    """
    Spatial mask readout with per-channel spatial masks.

    Unlike KlindtReadout2D which uses one shared spatial mask across all channels,
    this version has a separate spatial mask for each feature channel. This allows
    different features (e.g., ON vs OFF) to be pooled from different spatial locations,
    enabling the model to capture center-surround RF structure.
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
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.num_channels = list(num_kernels)[-1]  # Number of channels from core
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

        # Initialize per-channel mask weights
        # Shape: [H*W, num_channels, num_neurons] - one spatial mask per channel per neuron
        assert init_scales is not None, "init_scales must be provided"
        mean, std = init_scales[1]
        mask_init = torch.normal(
            mean=mean, std=std,
            size=(self.num_mask_pixels, self.num_channels, num_neurons)
        )
        self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights
        # Shape: [num_channels, num_neurons]
        mean, std = init_scales[2]
        self.readout_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(self.num_channels, num_neurons))
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
        assert C == self.num_channels, f"Input channels ({C}) must match num_channels ({self.num_channels})"

        # Flatten spatial dims
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # Apply per-channel spatial masks
        # x_flat: [B, C, H*W]
        # mask_weights: [H*W, C, N]
        # For each channel c: x_flat[:, c, :] @ mask_weights[:, c, :] -> [B, N]
        # Result: [B, C, N] where each channel used its own spatial mask

        masked = torch.zeros(B, C, self.num_neurons, device=x.device)
        for c in range(C):
            masked[:, c, :] = torch.matmul(x_flat[:, c, :], self.mask_weights[:, c, :])

        # Alternative efficient version using einsum (uncomment if needed):
        # masked = torch.einsum('bcp,pcn->bcn', x_flat, self.mask_weights)

        # Apply readout weights: [B, C, N] * [C, N] -> sum over C -> [B, N]
        output = (masked * self.readout_weights.unsqueeze(0)).sum(dim=1)

        if self.final_relu:
            output = F.softplus(output + self.bias)

        return output

    def regularizer(self) -> torch.Tensor:
        # Sum over spatial dims, mean over channels and neurons
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[0]
        weights_reg = torch.mean(torch.sum(torch.abs(self.readout_weights), dim=0)) * self.reg[1]
        return mask_reg + weights_reg


class KlindtReadoutNMasks2D(nn.Module):
    """
    Spatial mask readout with configurable number of spatial masks.

    Unlike KlindtReadoutPerChannel2D which has one mask per channel, this version
    allows specifying the number of masks independently. Each mask pools from ALL
    channels, then the pooled values are weighted and summed.

    For example, with 4 channels and 2 masks:
    - Mask 1 pools spatially from all 4 channels -> 4 values -> weighted sum -> scalar
    - Mask 2 pools spatially from all 4 channels -> 4 values -> weighted sum -> scalar
    - Final output = mask1_output + mask2_output

    This is useful for center-surround where you want:
    - Mask 1 = center location (pooling ON and OFF features from center)
    - Mask 2 = surround location (pooling ON and OFF features from surround)
    """

    def __init__(
        self,
        num_kernels: Iterable[int],
        num_neurons: int,
        num_masks: int,  # Number of spatial masks per neuron
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.num_channels = list(num_kernels)[-1]  # Number of channels from core
        self.num_masks = num_masks
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

        # Initialize spatial mask weights
        # Shape: [H*W, num_masks, num_neurons] - num_masks spatial masks per neuron
        assert init_scales is not None, "init_scales must be provided"
        mean, std = init_scales[1]
        mask_init = torch.normal(
            mean=mean, std=std,
            size=(self.num_mask_pixels, num_masks, num_neurons)
        )
        self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights
        # Shape: [num_masks, num_channels, num_neurons]
        # Each mask has weights for each channel
        mean, std = init_scales[2]
        self.readout_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(num_masks, self.num_channels, num_neurons))
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
                # Normalize over channels for each mask
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=1, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm
        elif self.weights_constraint == 'absnorm':
            with torch.no_grad():
                self.readout_weights.data = torch.abs(self.readout_weights.data)
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=1, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()

        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"
        assert C == self.num_channels, f"Input channels ({C}) must match num_channels ({self.num_channels})"

        # Flatten spatial dims
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # Apply spatial masks to get pooled values for each mask
        # x_flat: [B, C, H*W]
        # mask_weights: [H*W, M, N] where M = num_masks
        # For each mask m: sum over spatial (x_flat @ mask_weights[:, m, :]) -> [B, C, N]
        # Result: [B, M, C, N]

        output = torch.zeros(B, self.num_neurons, device=x.device)
        for m in range(self.num_masks):
            # Apply mask m to all channels: [B, C, H*W] @ [H*W, N] -> [B, C, N]
            masked = torch.matmul(x_flat, self.mask_weights[:, m, :])  # [B, C, N]
            # Weight channels and sum: [B, C, N] * [C, N] -> sum over C -> [B, N]
            weighted = (masked * self.readout_weights[m]).sum(dim=1)  # [B, N]
            output = output + weighted

        if self.final_relu:
            output = F.softplus(output + self.bias)

        return output

    def regularizer(self) -> torch.Tensor:
        # L1 on masks: sum over spatial, mean over masks and neurons
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[0]
        # L1 on weights: sum over channels, mean over masks and neurons
        weights_reg = torch.mean(torch.sum(torch.abs(self.readout_weights), dim=1)) * self.reg[1]
        return mask_reg + weights_reg


class KlindtCoreReadoutNMasks2D(nn.Module):
    """
    Complete Klindt-style CNN model with configurable number of spatial masks.

    This model allows specifying the number of spatial masks independently from
    the number of kernel channels. Each mask pools from all channels at a specific
    spatial location, enabling center-surround RF modeling with fewer parameters.
    """

    def __init__(
        self,
        # Core parameters
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        init_scales: Iterable[Iterable[float]],
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
        num_masks: int = 2,  # Number of spatial masks per neuron
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
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

        # Build readout with N masks
        self.readout = KlindtReadoutNMasks2D(
            num_kernels=num_kernels,
            num_neurons=num_neurons,
            num_masks=num_masks,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_scales=init_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()


class KlindtCoreONOFF2D(nn.Module):
    """
    Multi-layer 2D convolutional core with explicit ON/OFF kernel polarity.

    Kernels are split into two groups:
    - ON kernels: Constrained to have positive weights (detect light increments)
    - OFF kernels: Constrained to have negative weights (detect light decrements)
    """

    def __init__(
        self,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        n_on_kernels: int,
        n_off_kernels: int,
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

        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.num_kernels = n_on_kernels + n_off_kernels

        # Parse kernel sizes
        self.kernel_sizes = []
        for k in kernel_sizes:
            if isinstance(k, int):
                self.kernel_sizes.append((k, k))
            elif isinstance(k, Iterable) and len(k) == 2:
                self.kernel_sizes.append(tuple(k))
            else:
                raise ValueError(f"Invalid kernel size format: {k}")

        self.act_fns = list(act_fns)
        self.kernel_constraint = kernel_constraint

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Single conv layer with ON + OFF kernels
        k_size = self.kernel_sizes[0]
        conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=self.num_kernels,
            kernel_size=k_size,
            stride=1,
            padding=0,
        )

        # Weight initialization with ON/OFF polarity
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

                # Initialize ON kernels with positive Gaussian
                init_noise_on = torch.randn(n_on_kernels, image_channels, H, W)
                weight[:n_on_kernels].copy_(torch.abs(init_noise_on) * gaussian * init_scales[0][1])

                # Initialize OFF kernels with negative Gaussian
                init_noise_off = torch.randn(n_off_kernels, image_channels, H, W)
                weight[n_on_kernels:].copy_(-torch.abs(init_noise_off) * gaussian * init_scales[0][1])
        else:
            # Simple default initialization with polarity
            with torch.no_grad():
                nn.init.normal_(conv.weight[:n_on_kernels], mean=init_scales[0][0], std=init_scales[0][1])
                conv.weight[:n_on_kernels].data = torch.abs(conv.weight[:n_on_kernels].data)
                nn.init.normal_(conv.weight[n_on_kernels:], mean=init_scales[0][0], std=init_scales[0][1])
                conv.weight[n_on_kernels:].data = -torch.abs(conv.weight[n_on_kernels:].data)

        self.conv_layers.append(conv)

        if self.bn_layers is not None:
            self.bn_layers.append(
                nn.BatchNorm2d(
                    num_features=self.num_kernels,
                    affine=bn_cent,
                    track_running_stats=True,
                    momentum=0.02
                )
            )

        self.activation_layers.append(build_activation_layer(self.act_fns[0]))

        self.regularizer_module = L1Smooth2DRegularizer(
            sparsity_factor=sparsity_reg,
            smoothness_factor=smoothness_reg,
            center_mass_factor=center_mass_reg,
        )

    def apply_constraints(self):
        """Apply ON/OFF polarity constraints to kernels."""
        with torch.no_grad():
            conv = self.conv_layers[0]
            # ON kernels: ensure positive weights
            conv.weight[:self.n_on_kernels].data = torch.abs(conv.weight[:self.n_on_kernels].data)
            # OFF kernels: ensure negative weights
            conv.weight[self.n_on_kernels:].data = -torch.abs(conv.weight[self.n_on_kernels:].data)

            # Also apply norm constraint if specified
            if self.kernel_constraint == 'norm':
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


class KlindtReadoutONOFF2D(nn.Module):
    """
    Spatial mask readout with explicit ON/OFF pathway assignment.

    Two spatial masks per neuron:
    - Mask 0 (ON mask): Pools from ON kernel channels only
    - Mask 1 (OFF mask): Pools from OFF kernel channels only

    This allows clear visualization of which pathway (ON vs OFF) each cell uses.
    If a cell doesn't respond to ON pathway, its ON mask will be empty/zero.
    """

    def __init__(
        self,
        n_on_kernels: int,
        n_off_kernels: int,
        num_neurons: int,
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.num_channels = n_on_kernels + n_off_kernels
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

        # Initialize spatial mask weights
        # Shape: [H*W, 2, num_neurons] - 2 masks: ON and OFF
        assert init_scales is not None, "init_scales must be provided"
        mean, std = init_scales[1]
        mask_init = torch.normal(
            mean=mean, std=std,
            size=(self.num_mask_pixels, 2, num_neurons)
        )
        self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights for each pathway
        # ON pathway weights: [n_on_kernels, num_neurons]
        # OFF pathway weights: [n_off_kernels, num_neurons]
        mean, std = init_scales[2]
        self.on_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_on_kernels, num_neurons))
        )
        self.off_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_off_kernels, num_neurons))
        )

        # Bias for final nonlinearity
        self.bias = nn.Parameter(torch.full((num_neurons,), 0.5)) if final_relu else None

    def apply_constraints(self):
        if self.mask_constraint == 'abs':
            with torch.no_grad():
                self.mask_weights.data = torch.abs(self.mask_weights.data)

        if self.weights_constraint == 'abs':
            with torch.no_grad():
                self.on_weights.data = torch.abs(self.on_weights.data)
                self.off_weights.data = torch.abs(self.off_weights.data)
        elif self.weights_constraint == 'norm':
            with torch.no_grad():
                on_norm = torch.sqrt(torch.sum(self.on_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.on_weights.data = self.on_weights.data / on_norm
                off_norm = torch.sqrt(torch.sum(self.off_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.off_weights.data = self.off_weights.data / off_norm
        elif self.weights_constraint == 'absnorm':
            with torch.no_grad():
                self.on_weights.data = torch.abs(self.on_weights.data)
                on_norm = torch.sqrt(torch.sum(self.on_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.on_weights.data = self.on_weights.data / on_norm
                self.off_weights.data = torch.abs(self.off_weights.data)
                off_norm = torch.sqrt(torch.sum(self.off_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.off_weights.data = self.off_weights.data / off_norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()

        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"
        assert C == self.num_channels, f"Input channels ({C}) must match num_channels ({self.num_channels})"

        # Split features into ON and OFF channels
        on_features = x[:, :self.n_on_kernels, :, :]  # [B, n_on, H, W]
        off_features = x[:, self.n_on_kernels:, :, :]  # [B, n_off, H, W]

        # Flatten spatial dims
        on_flat = on_features.view(B, self.n_on_kernels, -1)  # [B, n_on, H*W]
        off_flat = off_features.view(B, self.n_off_kernels, -1)  # [B, n_off, H*W]

        # ON pathway: apply ON mask (mask 0) to ON features
        # on_flat: [B, n_on, H*W], mask_weights[:, 0, :]: [H*W, N]
        # Result: [B, n_on, N]
        on_masked = torch.matmul(on_flat, self.mask_weights[:, 0, :])  # [B, n_on, N]
        # Weight and sum: [B, n_on, N] * [n_on, N] -> sum over n_on -> [B, N]
        on_output = (on_masked * self.on_weights.unsqueeze(0)).sum(dim=1)  # [B, N]

        # OFF pathway: apply OFF mask (mask 1) to OFF features
        off_masked = torch.matmul(off_flat, self.mask_weights[:, 1, :])  # [B, n_off, N]
        off_output = (off_masked * self.off_weights.unsqueeze(0)).sum(dim=1)  # [B, N]

        output = on_output + off_output

        if self.final_relu:
            output = F.softplus(output + self.bias)

        return output

    def regularizer(self) -> torch.Tensor:
        # L1 on masks: sum over spatial, mean over masks and neurons
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[0]
        # L1 on weights: sum over channels, mean over neurons
        on_weights_reg = torch.mean(torch.sum(torch.abs(self.on_weights), dim=0)) * self.reg[1]
        off_weights_reg = torch.mean(torch.sum(torch.abs(self.off_weights), dim=0)) * self.reg[1]
        return mask_reg + on_weights_reg + off_weights_reg


class KlindtCoreReadoutONOFF2D(nn.Module):
    """
    Complete Klindt-style CNN model with explicit ON/OFF kernel polarity.

    Features:
    - ON kernels (positive weights): Detect light increments
    - OFF kernels (negative weights): Detect light decrements
    - ON spatial mask: Pools from ON kernels only
    - OFF spatial mask: Pools from OFF kernels only

    This architecture makes it explicit which pathway (ON vs OFF) each cell uses,
    and allows direct visualization of cell polarity preferences.
    """

    def __init__(
        self,
        # Core parameters
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        n_on_kernels: int = 2,
        n_off_kernels: int = 2,
        act_fns: Iterable[str] = ('relu',),
        init_scales: Iterable[Iterable[float]] = ((0.0, 0.01), (0.0, 0.001), (0.0, 0.01)),
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
        mask_reg: float = 1e-3,
        weights_reg: float = 1e-1,
    ):
        super().__init__()

        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels

        # Convert init_scales to numpy array
        init_scales = np.array(init_scales)

        # Build core with ON/OFF polarity
        self.core = KlindtCoreONOFF2D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
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

        # Build readout with ON/OFF pathway assignment
        self.readout = KlindtReadoutONOFF2D(
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_scales=init_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()


class KlindtReadoutONOFFMixed2D(nn.Module):
    """
    Spatial mask readout with ON, OFF, and Mixed pathways.

    Three spatial masks per neuron:
    - Mask 0 (ON mask): Pools from all ON kernel channels
    - Mask 1 (OFF mask): Pools from all OFF kernel channels
    - Mask 2 (Mixed mask): Pools from 1 ON + 1 OFF kernel (index 0 of each)

    The mixed mask allows cells to combine ON and OFF features at the same
    spatial location, which is needed for ON-OFF cells and polarity inversions.
    """

    def __init__(
        self,
        n_on_kernels: int,
        n_off_kernels: int,
        num_neurons: int,
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.num_channels = n_on_kernels + n_off_kernels
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

        # Initialize spatial mask weights
        # Shape: [H*W, 3, num_neurons] - 3 masks: ON, OFF, Mixed
        assert init_scales is not None, "init_scales must be provided"
        mean, std = init_scales[1]
        mask_init = torch.normal(
            mean=mean, std=std,
            size=(self.num_mask_pixels, 3, num_neurons)
        )
        self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights for ON and OFF pathways
        # ON pathway weights: [n_on_kernels, num_neurons]
        # OFF pathway weights: [n_off_kernels, num_neurons]
        mean, std = init_scales[2]
        self.on_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_on_kernels, num_neurons))
        )
        self.off_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_off_kernels, num_neurons))
        )

        # Mixed pathway weights: scalar weight for ON and OFF kernel (index 0)
        # Shape: [1, num_neurons] for each
        self.mixed_on_weight = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(1, num_neurons))
        )
        self.mixed_off_weight = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(1, num_neurons))
        )

        # Bias for final nonlinearity
        self.bias = nn.Parameter(torch.full((num_neurons,), 0.5)) if final_relu else None

    def apply_constraints(self):
        if self.mask_constraint == 'abs':
            with torch.no_grad():
                self.mask_weights.data = torch.abs(self.mask_weights.data)

        if self.weights_constraint == 'abs':
            with torch.no_grad():
                self.on_weights.data = torch.abs(self.on_weights.data)
                self.off_weights.data = torch.abs(self.off_weights.data)
                self.mixed_on_weight.data = torch.abs(self.mixed_on_weight.data)
                self.mixed_off_weight.data = torch.abs(self.mixed_off_weight.data)
        elif self.weights_constraint == 'norm':
            with torch.no_grad():
                on_norm = torch.sqrt(torch.sum(self.on_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.on_weights.data = self.on_weights.data / on_norm
                off_norm = torch.sqrt(torch.sum(self.off_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.off_weights.data = self.off_weights.data / off_norm
                # Mixed weights are scalars, normalize together
                mixed_norm = torch.sqrt(self.mixed_on_weight ** 2 + self.mixed_off_weight ** 2 + 1e-5)
                self.mixed_on_weight.data = self.mixed_on_weight.data / mixed_norm
                self.mixed_off_weight.data = self.mixed_off_weight.data / mixed_norm
        elif self.weights_constraint == 'absnorm':
            with torch.no_grad():
                self.on_weights.data = torch.abs(self.on_weights.data)
                on_norm = torch.sqrt(torch.sum(self.on_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.on_weights.data = self.on_weights.data / on_norm
                self.off_weights.data = torch.abs(self.off_weights.data)
                off_norm = torch.sqrt(torch.sum(self.off_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.off_weights.data = self.off_weights.data / off_norm
                # Mixed weights
                self.mixed_on_weight.data = torch.abs(self.mixed_on_weight.data)
                self.mixed_off_weight.data = torch.abs(self.mixed_off_weight.data)
                mixed_norm = torch.sqrt(self.mixed_on_weight ** 2 + self.mixed_off_weight ** 2 + 1e-5)
                self.mixed_on_weight.data = self.mixed_on_weight.data / mixed_norm
                self.mixed_off_weight.data = self.mixed_off_weight.data / mixed_norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()

        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"
        assert C == self.num_channels, f"Input channels ({C}) must match num_channels ({self.num_channels})"

        # Split features into ON and OFF channels
        on_features = x[:, :self.n_on_kernels, :, :]  # [B, n_on, H, W]
        off_features = x[:, self.n_on_kernels:, :, :]  # [B, n_off, H, W]

        # Flatten spatial dims
        on_flat = on_features.view(B, self.n_on_kernels, -1)  # [B, n_on, H*W]
        off_flat = off_features.view(B, self.n_off_kernels, -1)  # [B, n_off, H*W]

        # === ON pathway: apply ON mask (mask 0) to ON features ===
        # on_flat: [B, n_on, H*W], mask_weights[:, 0, :]: [H*W, N]
        on_masked = torch.matmul(on_flat, self.mask_weights[:, 0, :])  # [B, n_on, N]
        on_output = (on_masked * self.on_weights.unsqueeze(0)).sum(dim=1)  # [B, N]

        # === OFF pathway: apply OFF mask (mask 1) to OFF features ===
        off_masked = torch.matmul(off_flat, self.mask_weights[:, 1, :])  # [B, n_off, N]
        off_output = (off_masked * self.off_weights.unsqueeze(0)).sum(dim=1)  # [B, N]

        # === Mixed pathway: apply Mixed mask (mask 2) to first ON and first OFF kernel ===
        # Use only the first kernel (index 0) from each pathway
        mixed_on_flat = on_flat[:, 0:1, :]  # [B, 1, H*W]
        mixed_off_flat = off_flat[:, 0:1, :]  # [B, 1, H*W]

        # Apply mixed mask to both
        mixed_on_masked = torch.matmul(mixed_on_flat, self.mask_weights[:, 2, :])  # [B, 1, N]
        mixed_off_masked = torch.matmul(mixed_off_flat, self.mask_weights[:, 2, :])  # [B, 1, N]

        # Weight and sum
        mixed_on_output = (mixed_on_masked * self.mixed_on_weight.unsqueeze(0)).sum(dim=1)  # [B, N]
        mixed_off_output = (mixed_off_masked * self.mixed_off_weight.unsqueeze(0)).sum(dim=1)  # [B, N]
        mixed_output = mixed_on_output + mixed_off_output

        output = on_output + off_output + mixed_output

        if self.final_relu:
            output = F.softplus(output + self.bias)

        return output

    def regularizer(self) -> torch.Tensor:
        # L1 on masks: sum over spatial, mean over masks and neurons
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[0]
        # L1 on weights: sum over channels, mean over neurons
        on_weights_reg = torch.mean(torch.sum(torch.abs(self.on_weights), dim=0)) * self.reg[1]
        off_weights_reg = torch.mean(torch.sum(torch.abs(self.off_weights), dim=0)) * self.reg[1]
        # L1 on mixed weights
        mixed_on_reg = torch.mean(torch.abs(self.mixed_on_weight)) * self.reg[1]
        mixed_off_reg = torch.mean(torch.abs(self.mixed_off_weight)) * self.reg[1]
        return mask_reg + on_weights_reg + off_weights_reg + mixed_on_reg + mixed_off_reg


class KlindtCoreReadoutONOFFMixed2D(nn.Module):
    """
    Complete Klindt-style CNN model with ON/OFF kernel polarity and mixed pathway.

    Features:
    - ON kernels (positive weights): Detect light increments
    - OFF kernels (negative weights): Detect light decrements
    - ON spatial mask: Pools from ON kernels only
    - OFF spatial mask: Pools from OFF kernels only
    - Mixed spatial mask: Pools from 1 ON + 1 OFF kernel at the same location

    The mixed mask allows cells to combine both polarities at the same spatial
    location, enabling modeling of ON-OFF cells and polarity inversions.
    """

    def __init__(
        self,
        # Core parameters
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        n_on_kernels: int = 2,
        n_off_kernels: int = 2,
        act_fns: Iterable[str] = ('relu',),
        init_scales: Iterable[Iterable[float]] = ((0.0, 0.01), (0.0, 0.001), (0.0, 0.01)),
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
        mask_reg: float = 1e-3,
        weights_reg: float = 1e-1,
    ):
        super().__init__()

        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels

        # Convert init_scales to numpy array
        init_scales = np.array(init_scales)

        # Build core with ON/OFF polarity (same as KlindtCoreReadoutONOFF2D)
        self.core = KlindtCoreONOFF2D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
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

        # Build readout with ON/OFF/Mixed pathway assignment
        self.readout = KlindtReadoutONOFFMixed2D(
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_scales=init_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()


class KlindtCoreDedicatedONOFFMixed2D(nn.Module):
    """
    2D convolutional core with 6 dedicated kernels for ON/OFF/Mixed pathways.

    Kernel allocation:
    - Kernels 0-1: ON kernels (positive weights) → dedicated to ON mask
    - Kernels 2-3: OFF kernels (negative weights) → dedicated to OFF mask
    - Kernels 4-5: Mixed kernels (one ON, one OFF) → dedicated to Mixed mask

    Unlike KlindtCoreONOFF2D where kernels are shared between pathways, this
    architecture gives each pathway its own dedicated kernels. This prevents
    conflicting optimization pressures that can cause kernels to develop
    unexpected center-surround structures.
    """

    def __init__(
        self,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        n_on_kernels: int,
        n_off_kernels: int,
        n_mixed_kernels: int,
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

        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.n_mixed_kernels = n_mixed_kernels
        self.num_kernels = n_on_kernels + n_off_kernels + n_mixed_kernels

        # Parse kernel sizes
        self.kernel_sizes = []
        for k in kernel_sizes:
            if isinstance(k, int):
                self.kernel_sizes.append((k, k))
            elif isinstance(k, Iterable) and len(k) == 2:
                self.kernel_sizes.append(tuple(k))
            else:
                raise ValueError(f"Invalid kernel size format: {k}")

        self.act_fns = list(act_fns)
        self.kernel_constraint = kernel_constraint

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Single conv layer with all kernels
        k_size = self.kernel_sizes[0]
        conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=self.num_kernels,
            kernel_size=k_size,
            stride=1,
            padding=0,
        )

        # Weight initialization with dedicated polarity per pathway
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

                # Initialize ON kernels (0 to n_on-1) with positive Gaussian
                init_noise = torch.randn(n_on_kernels, image_channels, H, W)
                weight[:n_on_kernels].copy_(torch.abs(init_noise) * gaussian * init_scales[0][1])

                # Initialize OFF kernels (n_on to n_on+n_off-1) with negative Gaussian
                init_noise = torch.randn(n_off_kernels, image_channels, H, W)
                weight[n_on_kernels:n_on_kernels+n_off_kernels].copy_(
                    -torch.abs(init_noise) * gaussian * init_scales[0][1]
                )

                # Initialize Mixed kernels: first half ON-like, second half OFF-like
                n_mixed_on = n_mixed_kernels // 2
                n_mixed_off = n_mixed_kernels - n_mixed_on
                mixed_start = n_on_kernels + n_off_kernels

                # Mixed ON-like kernels
                init_noise = torch.randn(n_mixed_on, image_channels, H, W)
                weight[mixed_start:mixed_start+n_mixed_on].copy_(
                    torch.abs(init_noise) * gaussian * init_scales[0][1]
                )

                # Mixed OFF-like kernels
                init_noise = torch.randn(n_mixed_off, image_channels, H, W)
                weight[mixed_start+n_mixed_on:].copy_(
                    -torch.abs(init_noise) * gaussian * init_scales[0][1]
                )
        else:
            # Simple default initialization with polarity
            with torch.no_grad():
                # ON kernels
                nn.init.normal_(conv.weight[:n_on_kernels], mean=init_scales[0][0], std=init_scales[0][1])
                conv.weight[:n_on_kernels].data = torch.abs(conv.weight[:n_on_kernels].data)

                # OFF kernels
                nn.init.normal_(conv.weight[n_on_kernels:n_on_kernels+n_off_kernels],
                               mean=init_scales[0][0], std=init_scales[0][1])
                conv.weight[n_on_kernels:n_on_kernels+n_off_kernels].data = \
                    -torch.abs(conv.weight[n_on_kernels:n_on_kernels+n_off_kernels].data)

                # Mixed kernels: first half ON, second half OFF
                mixed_start = n_on_kernels + n_off_kernels
                n_mixed_on = n_mixed_kernels // 2
                nn.init.normal_(conv.weight[mixed_start:], mean=init_scales[0][0], std=init_scales[0][1])
                conv.weight[mixed_start:mixed_start+n_mixed_on].data = \
                    torch.abs(conv.weight[mixed_start:mixed_start+n_mixed_on].data)
                conv.weight[mixed_start+n_mixed_on:].data = \
                    -torch.abs(conv.weight[mixed_start+n_mixed_on:].data)

        self.conv_layers.append(conv)

        if self.bn_layers is not None:
            self.bn_layers.append(
                nn.BatchNorm2d(
                    num_features=self.num_kernels,
                    affine=bn_cent,
                    track_running_stats=True,
                    momentum=0.02
                )
            )

        self.activation_layers.append(build_activation_layer(self.act_fns[0]))

        self.regularizer_module = L1Smooth2DRegularizer(
            sparsity_factor=sparsity_reg,
            smoothness_factor=smoothness_reg,
            center_mass_factor=center_mass_reg,
        )

    def apply_constraints(self):
        """Apply polarity constraints to dedicated kernels."""
        with torch.no_grad():
            conv = self.conv_layers[0]

            # ON kernels (0 to n_on-1): positive weights
            if self.n_on_kernels > 0:
                conv.weight[:self.n_on_kernels].data = torch.abs(conv.weight[:self.n_on_kernels].data)

            # OFF kernels (n_on to n_on+n_off-1): negative weights
            if self.n_off_kernels > 0:
                off_start = self.n_on_kernels
                off_end = self.n_on_kernels + self.n_off_kernels
                conv.weight[off_start:off_end].data = -torch.abs(conv.weight[off_start:off_end].data)

            # Mixed kernels: first half ON (positive), second half OFF (negative)
            if self.n_mixed_kernels > 0:
                mixed_start = self.n_on_kernels + self.n_off_kernels
                n_mixed_on = self.n_mixed_kernels // 2
                if n_mixed_on > 0:
                    conv.weight[mixed_start:mixed_start+n_mixed_on].data = \
                        torch.abs(conv.weight[mixed_start:mixed_start+n_mixed_on].data)
                n_mixed_off = self.n_mixed_kernels - n_mixed_on
                if n_mixed_off > 0:
                    conv.weight[mixed_start+n_mixed_on:].data = \
                        -torch.abs(conv.weight[mixed_start+n_mixed_on:].data)

            # Apply norm constraint if specified
            if self.kernel_constraint == 'norm':
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


class KlindtReadoutDedicatedONOFFMixed2D(nn.Module):
    """
    Spatial mask readout with dedicated kernel assignment per pathway.

    Three spatial masks per neuron, each with its own dedicated kernels:
    - Mask 0 (ON mask): Pools only from ON kernels (indices 0 to n_on-1)
    - Mask 1 (OFF mask): Pools only from OFF kernels (indices n_on to n_on+n_off-1)
    - Mask 2 (Mixed mask): Pools only from Mixed kernels (indices n_on+n_off onwards)

    Unlike KlindtReadoutONOFFMixed2D where the Mixed mask shares kernels with ON/OFF,
    this readout gives each mask its own dedicated kernels. This prevents conflicting
    optimization pressures that can cause kernels to develop mixed polarity.
    """

    def __init__(
        self,
        n_on_kernels: int,
        n_off_kernels: int,
        n_mixed_kernels: int,
        num_neurons: int,
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_scales: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.n_mixed_kernels = n_mixed_kernels
        self.num_channels = n_on_kernels + n_off_kernels + n_mixed_kernels
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

        # Initialize spatial mask weights
        # Shape: [H*W, 3, num_neurons] - 3 masks: ON, OFF, Mixed
        assert init_scales is not None, "init_scales must be provided"
        mean, std = init_scales[1]
        mask_init = torch.normal(
            mean=mean, std=std,
            size=(self.num_mask_pixels, 3, num_neurons)
        )
        self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights for each pathway
        mean, std = init_scales[2]

        # ON pathway weights: [n_on_kernels, num_neurons]
        self.on_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_on_kernels, num_neurons))
        )

        # OFF pathway weights: [n_off_kernels, num_neurons]
        self.off_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_off_kernels, num_neurons))
        )

        # Mixed pathway weights: [n_mixed_kernels, num_neurons]
        self.mixed_weights = nn.Parameter(
            torch.normal(mean=mean, std=std, size=(n_mixed_kernels, num_neurons))
        )

        # Bias for final nonlinearity
        self.bias = nn.Parameter(torch.full((num_neurons,), 0.5)) if final_relu else None

    def _constrain_weights(self, w):
        """Apply weight constraint to a single weight tensor (skip if empty)."""
        if w.shape[0] == 0:
            return
        if self.weights_constraint == 'abs':
            w.data = torch.abs(w.data)
        elif self.weights_constraint == 'norm':
            norm = torch.sqrt(torch.sum(w ** 2, dim=0, keepdim=True) + 1e-5)
            w.data = w.data / norm
        elif self.weights_constraint == 'absnorm':
            w.data = torch.abs(w.data)
            norm = torch.sqrt(torch.sum(w ** 2, dim=0, keepdim=True) + 1e-5)
            w.data = w.data / norm

    def apply_constraints(self):
        if self.mask_constraint == 'abs':
            with torch.no_grad():
                self.mask_weights.data = torch.abs(self.mask_weights.data)

        with torch.no_grad():
            self._constrain_weights(self.on_weights)
            self._constrain_weights(self.off_weights)
            self._constrain_weights(self.mixed_weights)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()

        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"
        assert C == self.num_channels, f"Input channels ({C}) must match num_channels ({self.num_channels})"

        on_end = self.n_on_kernels
        off_end = on_end + self.n_off_kernels

        output = torch.zeros(B, self.num_neurons, device=x.device)

        # === ON pathway: apply ON mask (mask 0) to ON features only ===
        if self.n_on_kernels > 0:
            on_flat = x[:, :on_end, :, :].reshape(B, self.n_on_kernels, -1)
            on_masked = torch.matmul(on_flat, self.mask_weights[:, 0, :])  # [B, n_on, N]
            output = output + (on_masked * self.on_weights.unsqueeze(0)).sum(dim=1)

        # === OFF pathway: apply OFF mask (mask 1) to OFF features only ===
        if self.n_off_kernels > 0:
            off_flat = x[:, on_end:off_end, :, :].reshape(B, self.n_off_kernels, -1)
            off_masked = torch.matmul(off_flat, self.mask_weights[:, 1, :])  # [B, n_off, N]
            output = output + (off_masked * self.off_weights.unsqueeze(0)).sum(dim=1)

        # === Mixed pathway: apply Mixed mask (mask 2) to Mixed features only ===
        if self.n_mixed_kernels > 0:
            mixed_flat = x[:, off_end:, :, :].reshape(B, self.n_mixed_kernels, -1)
            mixed_masked = torch.matmul(mixed_flat, self.mask_weights[:, 2, :])  # [B, n_mixed, N]
            output = output + (mixed_masked * self.mixed_weights.unsqueeze(0)).sum(dim=1)

        if self.final_relu:
            output = F.softplus(output + self.bias)

        return output

    def regularizer(self) -> torch.Tensor:
        # L1 on masks: only regularize masks for active pathways
        active_mask_indices = []
        if self.n_on_kernels > 0:
            active_mask_indices.append(0)
        if self.n_off_kernels > 0:
            active_mask_indices.append(1)
        if self.n_mixed_kernels > 0:
            active_mask_indices.append(2)

        if active_mask_indices:
            mask_reg = torch.mean(torch.sum(
                torch.abs(self.mask_weights[:, active_mask_indices, :]), dim=0)) * self.reg[0]
        else:
            mask_reg = torch.tensor(0.0, device=self.mask_weights.device)

        # L1 on weights: only for active pathways
        weights_reg = torch.tensor(0.0, device=self.mask_weights.device)
        if self.n_on_kernels > 0:
            weights_reg = weights_reg + torch.mean(torch.sum(torch.abs(self.on_weights), dim=0)) * self.reg[1]
        if self.n_off_kernels > 0:
            weights_reg = weights_reg + torch.mean(torch.sum(torch.abs(self.off_weights), dim=0)) * self.reg[1]
        if self.n_mixed_kernels > 0:
            weights_reg = weights_reg + torch.mean(torch.sum(torch.abs(self.mixed_weights), dim=0)) * self.reg[1]

        return mask_reg + weights_reg


class KlindtCoreReadoutDedicatedONOFFMixed2D(nn.Module):
    """
    Complete Klindt-style CNN model with 6 dedicated kernels for ON/OFF/Mixed pathways.

    Architecture:
    - 2 ON kernels (positive weights): Dedicated to ON spatial mask
    - 2 OFF kernels (negative weights): Dedicated to OFF spatial mask
    - 2 Mixed kernels (1 ON-like, 1 OFF-like): Dedicated to Mixed spatial mask

    Unlike KlindtCoreReadoutONOFFMixed2D where ON/OFF kernels are shared with the
    Mixed pathway, this architecture gives each pathway its own dedicated kernels.
    This prevents conflicting optimization pressures that can cause kernels to
    develop unexpected center-surround structures.

    Key difference from shared architecture:
    - Shared: ON mask uses kernel 0,1; Mixed mask also uses kernel 0,2 (conflict!)
    - Dedicated: ON mask uses kernel 0,1; Mixed mask uses kernel 4,5 (no conflict)
    """

    def __init__(
        self,
        # Core parameters
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        n_on_kernels: int = 2,
        n_off_kernels: int = 2,
        n_mixed_kernels: int = 2,
        act_fns: Iterable[str] = ('relu',),
        init_scales: Iterable[Iterable[float]] = ((0.0, 0.01), (0.0, 0.001), (0.0, 0.01)),
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
        mask_reg: float = 1e-3,
        weights_reg: float = 1e-1,
    ):
        super().__init__()

        self.n_on_kernels = n_on_kernels
        self.n_off_kernels = n_off_kernels
        self.n_mixed_kernels = n_mixed_kernels

        # Convert init_scales to numpy array
        init_scales = np.array(init_scales)

        # Build core with dedicated ON/OFF/Mixed kernels
        self.core = KlindtCoreDedicatedONOFFMixed2D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            n_mixed_kernels=n_mixed_kernels,
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

        # Build readout with dedicated pathway assignment
        self.readout = KlindtReadoutDedicatedONOFFMixed2D(
            n_on_kernels=n_on_kernels,
            n_off_kernels=n_off_kernels,
            n_mixed_kernels=n_mixed_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_scales=init_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()


class KlindtCoreReadoutPerChannel2D(nn.Module):
    """
    Complete Klindt-style CNN model with per-channel spatial masks in the readout.

    This model allows different feature channels to be pooled from different spatial
    locations, enabling the capture of center-surround receptive field structure.
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

        # Build readout with per-channel masks
        self.readout = KlindtReadoutPerChannel2D(
            num_kernels=num_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=self.mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_scales=init_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.core.regularizer() + self.readout.regularizer()



