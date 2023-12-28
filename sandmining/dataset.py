import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio
from rasterio.windows import Window

import pickle as pkl

from .visualizations import visualize_raster_on_image
from project_config import IN_RIVER_BOUNDS_THRESHOLD

class PatchDataset(Dataset):
    def __init__(self, observation_directory, image_path, river_raster_path, labels_raster_path, transforms=None, patch_size=96):
        """
        Args:
            image_paths: Path to the image.
            transforms: Optional torchvision transforms to apply to image patches.
        """
        self.observation_directory = observation_directory
        self.observation_number = self.observation_directory.name[-1]
        self.image_path = image_path
        self.river_raster_path = river_raster_path
        self.labels_raster_path = labels_raster_path
        self.transforms = transforms
        self.patch_size = patch_size

        # Load the image using PIL. Save it as attribute.
        self.image = Image.open(image_path)
        self.image_width, self.image_height = self.image.size

        self.valid_indices = np.array([])
        self.cached_indices = self.observation_directory / f'valid_indices_obs{self.observation_number}.pkl'

    def __len__(self):
        # Brute force one time calculation to find valid indices.
        if (self.cached_indices).is_file():
            with open(self.cached_indices, 'rb') as f:
                self.valid_indices = pkl.load(f)
        if np.count_nonzero(self.valid_indices) == 0:
            max_num_patches = (self.image_height - self.patch_size + 1) * (self.image_width - self.patch_size + 1)
            valid_indices = set()
            self.valid_indices = set()
            possible_indices = set(np.arange(max_num_patches))
            while len(valid_indices) <= 50000: # Okay, since random sampling
                sampled_indices = np.random.random(20) * (len(possible_indices) - 1)
                sampled_indices = np.round(sampled_indices)
                sampled_indices = [int(i) for i in sampled_indices]
                try:
                    for i in sampled_indices:
                        if self.__getitem__(i):
                            valid_indices.add(i)
                except ValueError:
                    pass
                possible_indices.remove(i) if i not in possible_indices else None
                print(len(self.valid_indices))
                self.valid_indices = valid_indices
            with open(self.cached_indices, 'wb') as f:
                valid_indices = np.array(list(self.valid_indices))
                pkl.dump(valid_indices, f)
            
        return len(self.valid_indices)

    def __getitem__(self, idx):

        def find_nth_sliding_window_patch(idx):
            """
            Finds the coordinates of the nth sliding window patch in a rectangular image.

            Args:
                idx: Integer index representing the patch (0-based).
                image_height: Height of the image in pixels.
                image_width: Width of the image in pixels.
                patch_size: Size of the square sliding window patch (assumed equal width and height).

            Returns:
                A tuple containing the (x_min, y_min, x_max, y_max) coordinates of the patch.

            Raises:
                ValueError: If the patch size is larger than image dimensions or the index is out of bounds.
                            Or if the patch is not within the river bounds according to the threshold.
            """

            # Calculate the number of unique patches
            num_possible_patches = (self.image_height - self.patch_size + 1) * (self.image_width - self.patch_size + 1)

            # Validate index
            if idx >= num_possible_patches:
                raise ValueError("Index is out of bounds for the given image and patch size.")

            max_patches_in_a_row = self.image_width - self.patch_size + 1
            
            # Calculate (x, y) coordinates from the integer index
            left = idx % (max_patches_in_a_row - 1)
            upper = idx // (max_patches_in_a_row - 1)

            # Calculate remaining coordinates
            right = left + self.patch_size 
            lower = upper + self.patch_size 
            
            if (right >= self.image_width) or (lower >= self.image_height):
                print(f"Edge case: idx-{idx}, left-{left}, upper-{upper}, right-{right}, lower-{lower}")
                left -= 1
                right -= 1
                upper -= 1
                lower -= 1

            return left, upper, right, lower

        left, upper, right, lower = find_nth_sliding_window_patch(idx=idx)

        # Extract the patch using slicing or PyTorch's F.grid_sample
        patch = self.image.crop((int(left), int(upper), int(right), int(lower)))
        if patch.size != (96, 96):
            raise ValueError(f"Patch is the wrong size. {patch.size}")
        
        # Apply any image transformations if provided
        if self.transforms:
            patch = self.transforms(patch)
        # Calculate corresponding raster coordinates for the image patch
        window = Window(col_off=left, row_off=upper, width=self.patch_size, height=self.patch_size)

        # Read the raster file with rasterio
        with rasterio.open(self.labels_raster_path) as labels_src:
            # Get image-to-raster transformation
            labels_patch = labels_src.read(window=window)
            if labels_patch.shape != (1, 96, 96):
                raise ValueError(f"Labels patch wrong shape. {labels_patch.shape}")
            
        with rasterio.open(self.river_raster_path) as rivers_src:
            rivers_patch = rivers_src.read(window=window)

        # Note whether the patch is completely out of or decently within the river bounds
        if (np.count_nonzero(rivers_patch == 255) / (self.patch_size**2)) > IN_RIVER_BOUNDS_THRESHOLD:
            return (np.array(patch), labels_patch, [left, upper, right, lower])
        else:
            raise ValueError(f"Patch with index {idx} is not within river bounds.")