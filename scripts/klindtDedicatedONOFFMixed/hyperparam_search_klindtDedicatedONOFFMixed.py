import torch
import pickle
import os
from datetime import datetime
from center_surround.data import load_raw_data, create_dataloaders
from center_surround.training import run_hyperparameter_search_dedicated_onoff_mixed


EXP_NUM = 13

# Model name
model_name = "klindtDedicatedONOFFMixed"

# Select which cell type to train on
# cell_type = "reliability_bigger_than_0point5" # This can also be whatever batch of cells you wan to model
                                # You should just modify sellected_cell_ids to what ever you want
cell_type = "rel_bigger_0_3_cells_11kernels"
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
cell_type_for_modeling_file = f'exp_{EXP_NUM}_cell_types_for_modeling.pkl'
cell_types_path = f'/home/ridvan/Documents/center_surround/data/{cell_type_for_modeling_file}'
with open(cell_types_path, 'rb') as f:
    cell_types_data = pickle.load(f)

all_cells = cell_types_data['all_cells']
cell_type_dict = cell_types_data['cell_types']

# Select cell type to model
# selected_cell_ids = cell_type_dict[cell_type]
# selected_cell_ids = [16, 30, 31, 39, 54, 60, 112, 159, 173, 196, 205, 212, 229, 276,
#                         292, 331, 351, 370, 385, 405, 426, 428, 449, 559, 868,
#                         960, 997, 1103, 1118, 1168, 1293, 1356, 1368, 1370, 1418,
#                         1460, 1463, 1477, 1497, 1499, 1535, 1543]
# selected_cell_ids = [148,212,441,478,582,1054,1091,1122,1437,1441,1587]

selected_cell_ids = [
9, 23, 27, 30, 46, 63, 74, 75, 81, 87, 89, 90, 115, 122, 129, 137,
143, 144, 148, 169, 177, 187, 202, 206, 211, 212, 213, 221, 256,
277, 280, 287, 293, 304, 312, 330, 332, 342, 346, 348, 357, 359,
364, 398, 401, 402, 441, 468, 478, 488, 493, 497, 500, 507, 508,
534, 541, 542, 574, 580, 582, 614, 627, 630, 654, 684, 721, 723,
745, 755, 778, 789, 790, 864, 869, 882, 928, 984, 1004, 1011,
1036, 1046, 1053, 1054, 1059, 1073, 1087, 1091, 1109, 1122, 1138,
1156, 1262, 1278, 1311, 1317, 1324, 1393, 1413, 1425, 1433, 1437,
1441, 1443, 1493, 1499, 1506, 1529, 1538, 1554, 1559, 1583, 1587,
1604, 1657, 1662, 1669, 1695, 1705, 1707, 1731, 1734, 1735, 1742,
1743
]

# If you want to model specific cells, you can directly specify their IDs here instead of using the cell type dict. Just make sure they are in the same format as the IDs in all_cells (e.g. "cell_123").

if selected_cell_ids is not None:
    neuron_indices = [all_cells.index(cid) for cid in selected_cell_ids]
    picked_cells = selected_cell_ids
else:
    neuron_indices = None
    picked_cells = all_cells

# Load data
data_path_file = f"exp_{EXP_NUM}_full_data"
data_path = f"/home/ridvan/Documents/center_surround/data/{data_path_file}.pkl"
raw_data = load_raw_data(data_path)
dataloaders = create_dataloaders(raw_data, batch_size=32, normalize_images=True,
                                 neuron_indices=neuron_indices)

# Get shapes
images, responses = next(iter(dataloaders['train']))
num_neurons = responses.shape[1]
input_size = images.shape[2]
in_channels = images.shape[1]

print(f"Input: {in_channels} channels, {input_size}x{input_size}")
print(f"Neurons: {num_neurons}")

# Run hyperparameter search for Dedicated ON/OFF/Mixed model
print(f"\nStarting hyperparameter search for {model_name} model...")
print("Tuning: smoothness_reg, weights_reg, mask_reg, learning_rate")

results = run_hyperparameter_search_dedicated_onoff_mixed(
    dataloaders=dataloaders,
    input_size=input_size,
    in_channels=in_channels,
    num_neurons=num_neurons,
    device=device,
    n_trials=500,       # number of trials
    num_epochs=200,     # epochs per trial
    n_on_kernels=1,     # 2 ON kernels (dedicated to ON mask)
    n_off_kernels=1,    # 2 OFF kernels (dedicated to OFF mask)
    n_mixed_kernels=0,  # 2 Mixed kernels (dedicated to Mixed mask)
)

print("\n" + "="*50)
print("BEST RESULTS")
print("="*50)
print(f"Best validation correlation: {results['best_value']:.4f}")
print("\nBest hyperparameters:")
for key, value in results['best_params'].items():
    print(f"  {key}: {value}")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = f'/home/ridvan/Documents/center_surround/outputs/exp_{EXP_NUM}/{model_name}'

# Save results with timestamp
output_path_for_timestamped_results = f'{output_dir}/{cell_type}/run_hypSearch_{timestamp}'
os.makedirs(output_path_for_timestamped_results, exist_ok=True)
output_path = f'{output_path_for_timestamped_results}/exp_{EXP_NUM}_hyperparam_results_{timestamp}.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {output_path}")

# Also save as latest
latest_path = f'{output_dir}/exp_{EXP_NUM}_hyperparam_results_latest.pkl'
with open(latest_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Also saved to {latest_path}")
