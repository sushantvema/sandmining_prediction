import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from segmentation_models_pytorch import Unet
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from .dataset import PatchDataset
from project_config import MODELS_DIRECTORY, OBS_2_DIRECTORY
from .visualizations import visualize_raster_on_image

NUM_SAMPLES = 100 # There are 479 samples in the dataloader, but I haven't
                    # Figured out how to correctly implement the len function
THRESHOLD = 0.5

def evaluate_model():
    # Load the model (replace with your model loading code)
    model = Unet(in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(MODELS_DIRECTORY / "unet_dec22_1830pm_0_72.pth"))
    model.eval()  # Set model to evaluation mode

    # Extract all valid samples from the test image:
    dataset = PatchDataset(image_path=OBS_2_DIRECTORY / 'rgb.tif', river_raster_path=OBS_2_DIRECTORY / 'rivers_mask_obs2.tif',
                           labels_raster_path=OBS_2_DIRECTORY / 'labels_mask_obs2.tif')
    dataloader = DataLoader(dataset, batch_size=1)
    patches, labels, coordinates = sample_images(dataloader, num_samples=NUM_SAMPLES)

    # Create a batch of patches and shape them for model input
    patches_tensor = torch.stack([torch.from_numpy(item).float() for item in patches])
    patches_tensor = torch.stack([p.permute(2, 1, 0) for p in patches_tensor])

    # Perform inference
    with torch.no_grad():
        predictions = model(patches_tensor)

    # Process and visualize predictions 
    for i, patch in enumerate(patches):
        prediction = np.argmax(predictions[i].numpy(), axis=0)  # Assuming segmentation model
        prediction = predictions[i].numpy() >= THRESHOLD
        import ipdb; ipdb.set_trace()
        pred_img = visualize_raster_on_image(raster_patch=prediction, src_patch=patch)  # Implement your visualization function
        pred_img.save(OBS_2_DIRECTORY / f"pred_patch{i}.jpg")
        img_jpg = mpimg.imread(OBS_2_DIRECTORY / f"pred_patch{i}.jpg")
        plt.imshow(img_jpg)
        plt.show()

def sample_images(dataloader, num_samples):
    sampled_images = []
    sampled_labels = []
    sampled_coordinates = []
    counter = 0
    while counter <= num_samples:
        data = dataloader.dataset.__getitem__(counter)
        patch = data[0]
        labels = data[1]
        coordinates = data[2]
        sampled_images.append(patch)
        sampled_labels.append(labels)
        sampled_coordinates.append(coordinates)
        counter += 1
    return sampled_images, sampled_labels, sampled_coordinates