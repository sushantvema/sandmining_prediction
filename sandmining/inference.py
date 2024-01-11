import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from segmentation_models_pytorch import UnetPlusPlus
import torch
from torch.utils.data import DataLoader
import numpy as np

from .dataset import PatchDataset
from project_config import MODELS_DIRECTORY
from project_config import NUM_SAMPLES, PREDICTION_THRESHOLD, SKIP_N
from project_config import OBSERVATION_FOR_EVALUATION
from project_config import MODEL_TO_USE
from .visualizations import visualize_raster_on_image, visualize_binary_labels, create_mask_from_patches

import random

# # Check hardware acceleration for apple silicon
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")

def evaluate_model(predict_uniformly=False):
    # Announcements
    print("-------------")
    print(f"Observation to evaluate: {OBSERVATION_FOR_EVALUATION}")
    print(f"Uniform random sampling?: {predict_uniformly}")
    print(f"Model to use: {MODEL_TO_USE}")
    print(f"Prediction Threshold: {PREDICTION_THRESHOLD}")
    print(f"Number of samples to inference on: {NUM_SAMPLES}")
    print("-------------")

    # Extract all valid samples from the test image:
    observation_number = str(OBSERVATION_FOR_EVALUATION)[-1]
    dataset = PatchDataset(observation_directory=OBSERVATION_FOR_EVALUATION,
                        image_path=OBSERVATION_FOR_EVALUATION / 'rgb.tif', 
                        river_raster_path=OBSERVATION_FOR_EVALUATION / f'rivers_mask_obs{observation_number}.tif',
                        labels_raster_path=OBSERVATION_FOR_EVALUATION / f'labels_mask_obs{observation_number}.tif')
    len(dataset)
    
    # Load the model
    # model = UnetPlusPlus(in_channels=3, classes=1, activation='sigmoid', decoder_attention_type='scse')
    model = UnetPlusPlus(in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(MODELS_DIRECTORY / MODEL_TO_USE)) # Trained on Obs 0 and Obs 1 
    model.eval()  # Set model to evaluation mode

    if not predict_uniformly:
        dataloader = DataLoader(dataset, batch_size=1)
        num_valid_samples = len(dataset)
        patches_list, labels_list, coordinates_list = sample_images(dataloader, num_samples=NUM_SAMPLES, skip_n=SKIP_N)

        # Create a batch of patches and shape them for model input
        patches_tensor = torch.stack([torch.from_numpy(item).float() for item in patches_list])
        patches_tensor = torch.stack([p.permute(2, 1, 0) for p in patches_tensor])

        # Perform inference
        with torch.no_grad():
            predictions = model(patches_tensor)

        # Process and visualize predictions 
        for i in np.arange(NUM_SAMPLES):
            patch = patches_list[i]
            prediction = np.argmax(predictions[i].numpy(), axis=0)  # Assuming segmentation model
            prediction = predictions[i].numpy() >= PREDICTION_THRESHOLD
            prediction = torch.from_numpy(prediction).permute(2, 1, 0).numpy()
            pred_img = visualize_raster_on_image(raster_patch=prediction, src_patch=patch)  # Implement your visualization function
            pred_img.save(OBSERVATION_FOR_EVALUATION / f"pred_patch{i}.jpg")
            img_jpg = mpimg.imread(OBSERVATION_FOR_EVALUATION / f"pred_patch{i}.jpg")
            plt.imshow(img_jpg)
            plt.show()

            labels = labels_list[i]
            # labels_img = visualize_binary_labels(labels=labels)
            labels_img = Image.fromarray(labels[0] * 255, mode='L')
            plt.imshow(labels_img)
            plt.show()

            import ipdb; ipdb.set_trace()
            X = 1 + 2
    else:
        patches_list = []
        coordinates_list = []
        counter = 0
        for i in random.sample(list(dataset.valid_indices), NUM_SAMPLES):
            print(f"{(counter // 100) * 100} patches accumulated.") if counter % 100 == 0 else None
            patch, labels, coordinates = dataset.__getitem__(i)
            patches_list.append(patch)
            coordinates_list.append(tuple(coordinates))
            counter += 1

        # Create a batch of patches and shape them for model input
        patches_tensor = torch.stack([torch.from_numpy(item).float() for item in patches_list])
        patches_tensor = torch.stack([p.permute(2, 1, 0) for p in patches_tensor])

        print("-------------")
        print("Starting inference.")
        print("-------------")

        stitched_predictions_mask = create_mask_from_patches(model=model, patches=patches_tensor, 
                                 coordinates=coordinates_list, observation_img_dir=OBSERVATION_FOR_EVALUATION,
                                 prediction_threshold=PREDICTION_THRESHOLD)
        import ipdb; ipdb.set_trace()

def sample_images(dataloader, num_samples, skip_n):
    sampled_images = []
    sampled_labels = []
    sampled_coordinates = []
    
    sampled_indices = random.sample(list(dataloader.dataset.valid_indices), num_samples)
    # sampled_indices = np.round(sampled_indices)
    # sampled_indices = [int(i) for i in sampled_indices]
    
    for idx in sampled_indices:
        data = dataloader.dataset.__getitem__(idx)
        patch = data[0]
        labels = data[1]
        coordinates = data[2]
        sampled_images.append(patch)
        sampled_labels.append(labels)
        sampled_coordinates.append(coordinates)
    return sampled_images, sampled_labels, sampled_coordinates
    