import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms  # For data augmentation (optional)
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from segmentation_models_pytorch import Unet

# Import custom dataset for sampling
from .dataset import PatchDataset

NUM_CHANNELS = 3
NUM_CLASSES = 1

def train_model(model_name, epoch=1):
    # list of candidate models and their parameters
    models = {
        "unet": Unet,
        "fcn": fcn_resnet50,
        "deeplabv3": deeplabv3_resnet50,
    }

    model = models[model_name]
    model = model(in_channels=NUM_CHANNELS, classes=NUM_CLASSES)  # Adjust channels for RGB input and binary output

    # Define loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data
    dataset = PatchDataset(image_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/rgb.tif",
                       river_raster_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/rivers_mask_obs0.tif",
                       labels_raster_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/labels_mask_obs0.tif")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch size
    # Training loop
    for epoch in range(1):
        for src_patch, labels_patch, in_river_bounds in dataloader:
            import ipdb; ipdb.set_trace()
            # Make sure patches are within the river bounds
            if in_river_bounds:
                # Forward pass
                outputs = model(src_patch)
                loss = loss_fn(outputs, labels_patch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print epoch loss
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save the trained model (optional)
    torch.save(model.state_dict(), "{}.pth".format(model_name))