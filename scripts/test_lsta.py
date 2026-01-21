import torch
from center_surround.utils import compute_lsta, plot_lsta, plot_mean_lsta
from torch.utils.data import DataLoader

# Load your trained model
model = torch.load_state_dict(torch.load('outputs/run_XXXX/best_model.pth'))

# Get some test images
test_images = DataLoader['test'].dataset.images

# Compute LSTA
lsta = compute_lsta(model, test_images, device='cuda')

# Plot
fig = plot_mean_lsta(lsta)
plt.show()