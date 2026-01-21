import torch
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.models import KlindtCoreReadout2D

# Load data
data_path = "data/exp_13_data.pkl"
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

# Create model
model = KlindtCoreReadout2D(
    image_size=input_size,
    image_channels=in_channels,
    kernel_sizes=[31, 31],
    num_kernels=[8, 16],
    act_fns=['relu', 'relu'],
    init_scales=[[0.0, 0.1], [0.0, 0.01], [0.0, 0.01]],
    num_neurons=num_neurons,
    smoothness_reg=1e0,
    sparsity_reg=1e-1,
    dropout_rate=0.2,
    final_relu=True,
)

print(f"\nModel created successfully!")
print(f"Mask size: {model.mask_size}")

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

print("\nAll tests passed!")