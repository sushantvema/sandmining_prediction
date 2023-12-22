import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio
from rasterio.windows import Window

from .visualizations import visualize_raster_on_image

class PatchDataset(Dataset):
    def __init__(self, image_path, river_raster_path, labels_raster_path, transforms=None, patch_size=96):
        """
        Args:
            image_paths: Path to the image.
            transforms: Optional torchvision transforms to apply to image patches.
        """
        self.image_path = image_path
        self.river_raster_path = river_raster_path
        self.labels_raster_path = labels_raster_path
        self.transforms = transforms
        self.patch_size = patch_size

        # Load the image using PIL. Save it as attribute.
        self.image = Image.open(image_path)
        self.image_width, self.image_height = self.image.size

    def __len__(self):
        # Calculate total number of patches across image (adjust here based on specific sampling strategy)
        num_unique_patches = (self.image_height - self.patch_size + 1) * (self.image_width - self.patch_size + 1)
        return num_unique_patches

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
            """

            # Calculate the number of unique patches
            num_patches = (self.image_height - self.patch_size + 1) * (self.image_width - self.patch_size + 1)

            # Validate index
            if idx >= num_patches:
                raise ValueError("Index is out of bounds for the given image and patch size.")

            max_patches_in_a_row = self.image_width - self.patch_size + 1
            
            # Calculate (x, y) coordinates from the integer index
            left = idx % max_patches_in_a_row 
            upper = idx // max_patches_in_a_row

            # Calculate remaining coordinates
            right = left + self.patch_size - 1
            lower = upper + self.patch_size - 1

            return left, upper, right, lower

        left, upper, right, lower = find_nth_sliding_window_patch(idx=idx)

        # Extract the patch using slicing or PyTorch's F.grid_sample
        patch = self.image.crop((left, upper, right, lower))

        # Apply any image transformations if provided
        if self.transforms:
            patch = self.transforms(patch)

        # Calculate corresponding raster coordinates for the image patch
        window = Window(col_off=left, row_off=upper, width=self.patch_size, height=self.patch_size)

        # Read the raster file with rasterio
        with rasterio.open(self.labels_raster_path) as labels_src:
            # Get image-to-raster transformation
            labels_patch = labels_src.read(window=window)
        with rasterio.open(self.river_raster_path) as rivers_src:
            rivers_patch = rivers_src.read(window=window)

        # Note whether the patch is completely or partially within the river bounds
        in_river_bounds = np.any(rivers_patch == 0)

        if in_river_bounds:
            return (np.array(patch), labels_patch)
        return None

# Example usage
# dataset = PatchDataset(image_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/rgb.tif",
#                        river_raster_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/rivers_mask_obs0.tif",
#                        labels_raster_path="/Users/sashikanth/Documents/sushi/sushi_personal/sandmining_prediction/sandmining/data/Observation0/labels_mask_obs0.tif")
# ds_length = len(dataset)
# random_patch_idxs = [(0, 0), (dataset.image_width - dataset.patch_size, dataset.image_height - dataset.patch_size)]
# import ipdb; ipdb.set_trace()
# for idx in random_patch_idxs:
#     return_tuple = dataset.__getitem__(idx=idx)
#     src_patch = return_tuple[0]
#     labels_patch = return_tuple[1]
#     in_river_bounds = return_tuple[2]
#     labels_on_src_image = visualize_raster_on_image(raster_patch=labels_patch, src_patch=src_patch)
    


# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch in dataloader:
#     image, labels = batch['image'], batch['label']
#     # ... further processing ...
