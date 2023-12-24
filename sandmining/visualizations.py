import matplotlib.pyplot as plt  # For interactive plotting and saving
import numpy as np
from PIL import Image
from PIL import ImageFilter

def visualize_raster_on_image(raster_patch, src_patch):
    """
    Visualizes a boolean raster patch on top of an RGB image patch with opaque overlay and bold borders.
    Assumes both patches have the same coordinates and therefore the same dimensions.

    Args:
        raster_image (np.array): RGB image in the form of an np array
        src_image (np.array): Raster image in the form of an np array

    Returns:
        Image: The combined image with the raster overlay.
    """
    raster_patch = raster_patch.astype(np.uint8)  # Convert to uint8 for compatibility

    # Create opaque overlay with bold borders:
    overlay = Image.fromarray(
        np.where(raster_patch, np.full((96, 96, 3), (0, 0, 255)), np.full((96, 96, 3), (0, 0, 0))),
        mode="RGB"
    )

    if not np.count_nonzero(np.array(overlay)):
        overlay = overlay.filter(ImageFilter.FIND_EDGES)  # Highlight edges
        overlay = overlay.point(lambda x: x * 4)  # Thicken edges

    # Blend overlay onto image:
    src_patch_to_image = Image.fromarray(src_patch, 'RGB')
    src_with_overlay = Image.blend(src_patch_to_image, overlay, alpha=0.5)  # Semi-transparent for visibility
    return src_with_overlay

def visualize_binary_labels_on_rgb_image(
    rgb_image_path,
    binary_labels_patches,
    coordinates,
    target_directory,
):
    """
    Visualizes binary label patches on top of an RGB image.

    Args:
        rgb_image_path (str): Path to the RGB image file.
        binary_labels_patches (list): List of binary label patches (as NumPy arrays).
        coordinates (list): List of tuples (x0, y0, x1, y1) representing the coordinates
            of each patch on the RGB image.
        target_directory (str, optional): Directory to save the visualization.
            Defaults to "visualizations".
    """

    # 1. Load RGB image
    rgb_image = Image.open(rgb_image_path)

    # 2. Create a combined image with labels overlaid
    combined_image = create_combined_image(rgb_image, binary_labels_patches, coordinates)

    # 3. Plot the combined image interactively
    plt.imshow(combined_image)
    plt.show()

    # 4. Save the combined image to the target directory
    save_combined_image(combined_image, target_directory)


def create_combined_image(rgb_image, binary_labels_patches, coordinates):
    """
    Combines the RGB image with binary labels overlaid on their respective coordinates.
    Handles potential alpha channels and color mapping for clarity.
    """
    # # Zip the tensors together, extract integer values using .item(), and create tuples
    # coordinates = list(zip(*[coords.tolist() for coords in zip(*coordinates)]))  # Transpose and convert to lists

    rgb_array = np.array(rgb_image)  # Convert RGB image to NumPy array
    combined_image = rgb_array.copy()  # Create a copy to avoid modifying the original

    for label_patch, (x0, y0, x1, y1) in zip(binary_labels_patches, coordinates):
        label_patch = np.array(label_patch)  # Ensure label patch is a NumPy array

        # Handle potential alpha channels:
        if label_patch.shape[-1] == 4:
            # Extract alpha channel for smooth blending:
            alpha_channel = label_patch[:, :, 3]
            label_patch = label_patch[:, :, :3]  # Remove alpha channel from color data
        else:
            alpha_channel = None  # No alpha channel to handle

        # Overlay label patch onto the RGB image:
        combined_image[y0:y1, x0:x1] = overlay_images(
            rgb_array[y0:y1, x0:x1], label_patch, alpha_channel
        )

    return Image.fromarray(combined_image)  # Convert back to PIL Image


def overlay_images(rgb_patch, label_patch, alpha_channel=None):
    """
    Overlays a label patch onto an RGB patch, optionally blending with an alpha channel.
    """

    if alpha_channel is not None:
        # Blend label patch with alpha channel for smooth transitions:
        return (rgb_patch * (1 - alpha_channel) + label_patch * alpha_channel).astype(np.uint8)

    else:
        # Simple color mapping for binary labels (e.g., red for clarity):
        label_patch = np.where(label_patch > 0, np.array([[255, 0, 0]] * 96), label_patch)  # Red color for labels
        return np.dstack((rgb_patch, label_patch)).astype(np.uint8)  # Combine channels


def save_combined_image(combined_image, target_directory):
    """
    Saves the combined image to the target directory in a suitable format.
    """
    return
    # ... (Implementation for saving the image, handling file extensions, etc.)
    combined_image.save("{}/combined.tif".format(target_directory))

