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


# import pickle
# with open('/home/ridvan/Documents/center_surround/data/exp_13_data.pkl', 'rb') as f:
#     data = pickle.load(f)

# for key, val in data.items():
#     print(f"{key}: {val.shape}")

import pickle
import numpy as np

with open('/home/ridvan/Documents/center_surround/data/exp_13_data_1.pkl', 'rb') as f:
    data = pickle.load(f)

print("=== SHAPES ===")
for key in data:
    if hasattr(data[key], 'shape'):
        print(f"{key}: {data[key].shape}")

print("\n=== IMAGE STATISTICS ===")
print(f"Train images - mean: {data['images_train'].mean():.4f}, std: {data['images_train'].std():.4f}")
print(f"Val images   - mean: {data['images_val'].mean():.4f}, std: {data['images_val'].std():.4f}")
print(f"Test images  - mean: {data['images_test'].mean():.4f}, std: {data['images_test'].std():.4f}")

print("\n=== RESPONSE STATISTICS ===")
print(f"Train resp - mean: {data['responses_train'].mean():.4f}, std: {data['responses_train'].std():.4f}")
print(f"Val resp   - mean: {data['responses_val'].mean():.4f}, std: {data['responses_val'].std():.4f}")
print(f"Test resp  - mean: {data['responses_test'].mean():.4f}, std: {data['responses_test'].std():.4f}")

print("\n=== TEST DATA SAMPLES ===")
print(f"Test images unique values (first 10): {np.unique(data['images_test'].flatten())[:10]}")
print(f"Test images min/max: {data['images_test'].min():.4f} / {data['images_test'].max():.4f}")