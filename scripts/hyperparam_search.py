import torch
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.training import run_hyperparameter_search

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
data_path = "data/exp_13_data_1.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32, normalize_images=True)
# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"Input: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Run hyperparameter search
print("\nStarting hyperparameter search...")
results = run_hyperparameter_search(
    dataloaders=dataloaders,
    input_size=input_size,
    in_channels=in_channels,
    num_neurons=num_neurons,
    device=device,
    n_trials=400,      # number of trials
    num_epochs=100,    # epochs per trial
)

print("\n" + "="*50)
print("BEST RESULTS")
print("="*50)
print(f"Best validation correlation: {results['best_value']:.4f}")
print("\nBest hyperparameters:")
for key, value in results['best_params'].items():
    print(f"  {key}: {value}")

# Save results
import pickle
with open('/home/ridvan/Documents/center_surround/outputs/hyperparam_results_1.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to outputs/hyperparam_results_1.pkl")
