import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms  # For data augmentation (optional)
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Import custom dataset for sampling
from .dataset import PatchDataset

# Config variables
from project_config import MODELS_DIRECTORY, OBSERVATIONS_FOR_TRAINING
from project_config import NUM_EPOCHS, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE, SAMPLES_TO_TRAIN_PER_OBSERVATION
from project_config import SAMPLES_PER_SAVE_WEIGHTS

import datetime
import time

# PyTorch implementation of DiceLoss (region-based loss function)
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train_model(model_name, epoch=1):
    # list of candidate models and their parameters
    models = {
        "unetplusplus": UnetPlusPlus,
        "fcn": fcn_resnet50,
        "deeplabv3": deeplabv3_resnet50,
    }

    model = models[model_name]
    model = model(in_channels=NUM_CHANNELS, classes=NUM_CLASSES)  # Adjust channels for RGB input and binary output

    # Define loss function and optimizer
    # loss_fn = nn.BCEWithLogitsLoss()  # For binary segmentation
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data
    for idx, OBSERVATION in enumerate(OBSERVATIONS_FOR_TRAINING):
        print("Start training with observation {}".format(idx))
        image_path = OBSERVATION / 'rgb.tif'
        river_raster_path = OBSERVATION / 'rivers_mask_obs{}.tif'.format(idx)
        labels_raster_path = OBSERVATION / 'labels_mask_obs{}.tif'.format(idx)

        dataset = PatchDataset(image_path = image_path,
                               observation_directory=OBSERVATION,
                        river_raster_path = river_raster_path,
                        labels_raster_path = labels_raster_path)
        num_valid_samples = len(dataset)
        sampler = SubsetRandomSampler(indices=list(dataset.valid_indices))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            for idx2, (src_patch, labels_patch, coordinates) in enumerate(dataloader):
                if (idx2 * 4) // SAMPLES_TO_TRAIN_PER_OBSERVATION >= 1:
                    break
                # Forward pass
                reshaped_src_patch = src_patch.permute(0, 3, 1, 2)  # Reshape to (4, 3, 96, 96)
                reshaped_src_patch = reshaped_src_patch.float()
                outputs = model(reshaped_src_patch)
                loss = loss_fn(outputs, labels_patch.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Processed {idx2*4} samples out of {len(dataset)} valid samples. ", end="\r")

                if idx2*4 % SAMPLES_PER_SAVE_WEIGHTS == 0:
                    # Save the trained model (optional)
                    print("\nSaving intermediate model")
                    timestamp = datetime.date.fromtimestamp(time.time()).__str__()
                    dest = MODELS_DIRECTORY / "{}_{}.pth".format(model_name, f"{timestamp}_{idx}_{idx2}_{int(time.time())}")
                    torch.save(model.state_dict(), dest)

            # Print epoch loss
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")