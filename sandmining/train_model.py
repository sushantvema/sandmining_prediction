import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms  # For data augmentation (optional)
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Import custom dataset for sampling
from .dataset import PatchDataset
from .visualizations import visualize_binary_labels_on_rgb_image
from project_config import OBS_0_DIRECTORY, OBS_1_DIRECTORY, OBS_2_DIRECTORY, MODELS_DIRECTORY

NUM_CHANNELS = 3
NUM_CLASSES = 1
BATCH_SIZE = 4
SAMPLING_FRACTION = 0.25

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
    observations_for_training = [OBS_0_DIRECTORY, OBS_1_DIRECTORY]
    for idx, OBSERVATION in enumerate(observations_for_training):
        print("Start training with observation {}".format(idx))
        image_path = OBSERVATION / 'rgb.tif'
        river_raster_path = OBSERVATION / 'rivers_mask_obs{}.tif'.format(idx)
        labels_raster_path = OBSERVATION / 'labels_mask_obs{}.tif'.format(idx)

        dataset = PatchDataset(image_path = image_path,
                        river_raster_path = river_raster_path,
                        labels_raster_path = labels_raster_path)
        
        # Create a sampler that selects a random subset of indices
        # sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:int(SAMPLING_FRACTION * len(dataset))])

        # Not the best approach. Filters out samples that are out of river bounds.
        #   and makes sure that samples are only used once. 
        used_indices = set()  # Track used sample indices
        def custom_collate_fn(batch):
            valid_items = [item for item in batch if item is not None]
            while len(valid_items) < BATCH_SIZE:
                more_indices = torch.randint(len(dataloader.dataset), size=(BATCH_SIZE - len(valid_items),))
                more_indices = [int(i) for i in more_indices if i not in used_indices]  # Filter out used ones
                used_indices.update(more_indices)  # Update used indices
                more_items = [dataloader.dataset[i] for i in more_indices]
                valid_items.extend([item for item in more_items if item is not None])
            return torch.utils.data.dataloader.default_collate(valid_items)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

        # Training loop
        for epoch in range(1):
            for idx2, (src_patch, labels_patch, coordinates) in enumerate(dataloader):
                print(f"Processed {idx2*4} samples out of {len(dataset)}", end="\r")
                # visualize_binary_labels_on_rgb_image(rgb_image_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/rgb.tif",
                #                                      binary_labels_patches=labels_patch, coordinates=coordinates, target_directory=None)
                # Forward pass
                reshaped_src_patch = src_patch.permute(0, 3, 1, 2)  # Reshape to (4, 3, 96, 96)
                reshaped_src_patch = reshaped_src_patch.float()
                outputs = model(reshaped_src_patch)
                loss = loss_fn(outputs, labels_patch.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if idx2*4 % 96 == 0:
                    # Save the trained model (optional)
                    print("\nSaving intermediate model")
                    dest = MODELS_DIRECTORY / "{}_{}.pth".format(model_name, f"dec22_1830pm_{idx}_{idx2}")
                    torch.save(model.state_dict(), dest)

            # Print epoch loss
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")