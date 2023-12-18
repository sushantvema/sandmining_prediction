import pandas as pd
import rasterio
from rasterio.features import rasterize

from project_config import DATA_DIRECTORY

import os

def process_annotations(data_directory):
    """
    For each of the observations in the data directory, get the height and the width
    as well as the file path for the .tif RGB image. 
        Step 1: Rasterize the annotations and rivers.
        Step 2: Save them the data directory.
        Step 3: Visualize the rasterized shapes on top of the RGB image. 
    """
    for observation_dir in os.listdir(DATA_DIRECTORY):
        metadata = pd.read
    return

def rasterize_annotation(geojson_file, target_dir):
    # Load GeoJSON data
    with open("your_file.geojson", "r") as f:
        geojson_data = f.read()

    # Define output raster parameters
    driver = "GTiff"
    height, width = 500, 500  # Adjust these to your desired image size
    crs = "epsg:4326"  # Change to your coordinate reference system
    dtype = "uint8"  # Data type for your raster values

    # Create an empty raster
    dst_profile = rasterio.profile.Profile(
        driver=driver,
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=[0, 1, 0, 0, -1, 0],
    )
    with rasterio.open("output.tif", "w", **dst_profile) as dst:
        pass
        
    rasterize(
        geojson_data,
        out_data=dst,
        fill=0,  # Background value for areas outside features
        driver=driver,
        dtype=dtype,
        transform=[0, 1, 0, 0, -1, 0],
    )

    dst.close()