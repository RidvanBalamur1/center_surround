import torch
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.models.klindtSurround import KlindtCoreReadoutPerChannel2D

# Load data
data_path = "data/exp_13_data_4.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32)

# Get a batch to check shapes
images, responses = next(iter(dataloaders['train']))
print(f"Images shape: {images.shape}")
print(f"Responses shape: {responses.shape}")

# Model parameters
num_neurons = responses.shape[1]
input_size = images.shape[2]  # assuming square
in_channels = images.shape[1]
num_kernels = [4]  # 4 channels = 4 spatial masks per neuron

# Create model
model = KlindtCoreReadoutPerChannel2D(
    image_size=input_size,
    image_channels=in_channels,
    kernel_sizes=[24],
    num_kernels=num_kernels,
    act_fns=['relu'],
    init_scales=[[0.0, 0.01], [0.0, 0.001], [0.0, 0.01]],
    num_neurons=num_neurons,
    smoothness_reg=1e0,
    sparsity_reg=1e-1,
    dropout_rate=0.2,
    final_relu=True,
    mask_constraint='abs',
    weights_constraint='abs',
)

print(f"\nModel created successfully!")
print(f"Mask size: {model.mask_size}")
print(f"Number of channels (spatial masks per neuron): {num_kernels[-1]}")

# Check mask_weights shape
print(f"Mask weights shape: {model.readout.mask_weights.shape}")
print(f"  Expected: [H*W={model.mask_size[0]*model.mask_size[1]}, num_channels={num_kernels[-1]}, num_neurons={num_neurons}]")

# Test forward pass
output = model(images)
print(f"Output shape: {output.shape}")

# Test regularizer
reg = model.regularizer()
print(f"Regularizer: {reg.item():.4f}")

# Test gradient flow
loss = output.mean() + reg
loss.backward()
print("Gradient flow: OK")

# Check that gradients exist for mask_weights
print(f"Mask weights gradient exists: {model.readout.mask_weights.grad is not None}")

print("\nAll tests passed!")