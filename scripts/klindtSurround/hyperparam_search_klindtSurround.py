import torch
import pickle
import os
from datetime import datetime
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.training import run_hyperparameter_search_surround

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
data_path_file = "exp_13_data_4"
data_path = f"/home/ridvan/Documents/center_surround/data/{data_path_file}.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32, normalize_images=True)

# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"Input: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Run hyperparameter search for surround model
print("\nStarting hyperparameter search for KlindtSurround model...")
print("Tuning: smoothness_reg, sparsity_reg, weights_reg, mask_reg, learning_rate, init_kernels_sigma")
results = run_hyperparameter_search_surround(
    dataloaders=dataloaders,
    input_size=input_size,
    in_channels=in_channels,
    num_neurons=num_neurons,
    device=device,
    n_trials=100,      # number of trials
    num_epochs=100,     # epochs per trial (shorter for search)
)

print("\n" + "="*50)
print("BEST RESULTS")
print("="*50)
print(f"Best validation correlation: {results['best_value']:.4f}")
print("\nBest hyperparameters:")
for key, value in results['best_params'].items():
    print(f"  {key}: {value}")

# Create output directory
output_dir = f'/home/ridvan/Documents/center_surround/outputs/{data_path_file}/klindtSurround'
os.makedirs(output_dir, exist_ok=True)

# Save results with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'{output_dir}/hyperparam_results_{timestamp}.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {output_path}")

# Also save as latest
latest_path = f'{output_dir}/hyperparam_results_latest.pkl'
with open(latest_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Also saved to {latest_path}")
