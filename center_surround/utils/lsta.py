
import torch
import numpy as np


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