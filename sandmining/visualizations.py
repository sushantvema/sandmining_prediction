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
        np.where(raster_patch[0][..., None], np.full((96, 96, 3), (0, 0, 255)), np.full((96, 96, 3), (0, 0, 0))),
        mode="RGB"
    )

    if not np.count_nonzero(np.array(overlay)):
        overlay = overlay.filter(ImageFilter.FIND_EDGES)  # Highlight edges
        overlay = overlay.point(lambda x: x * 4)  # Thicken edges

    # Blend overlay onto image:
    src_patch_to_image = Image.fromarray(src_patch, 'RGB')
    src_with_overlay = Image.blend(src_patch_to_image, overlay, alpha=0.5)  # Semi-transparent for visibility
    return src_with_overlay
