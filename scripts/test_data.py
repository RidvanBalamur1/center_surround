# import torch
# from center_surround.data import load_raw_data, create_dataloaders
# from center_surround.models import LNModel
# from center_surround.training import train

# # Load data
# raw_data = load_raw_data("data/exp_13_data.pkl")
# dataloaders = create_dataloaders(raw_data, batch_size=32)

# # Create model
# num_neurons = 8
# model = LNModel(num_neurons=num_neurons)

# # Train
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = train(model, dataloaders, num_epochs=5, lr=0.001, device=device)

# print("Training complete!")


# from center_surround.utils import compute_metrics

# # Evaluate on test set
# metrics = compute_metrics(model, dataloaders['test'], device)

# print("\nTest Results:")
# print(f"  MSE: {metrics['mse']:.4f}")
# print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
# print(f"  Per-neuron Correlation: {metrics['correlation_per_neuron']}")


import pickle
with open('/home/ridvan/Documents/center_surround/data/exp_13_data.pkl', 'rb') as f:
    data = pickle.load(f)

for key, val in data.items():
    print(f"{key}: {val.shape}")