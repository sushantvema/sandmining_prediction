from PIL import Image
from PIL.ExifTags import TAGS

import csv
import glob
import json
import os
import pathlib
import requests

with open('sandmining/data/observations.json', 'r') as f:
    observations_json = json.load(f)

def load_observations(observations_json=observations_json):
    for idx, observation in enumerate(observations_json):
        parent_dir = os.getcwd()
        target_dir = parent_dir + '/sandmining/data/Observation{}'.format(idx)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        
        uri_to_s2 = observation['uri_to_s2']
        uri_to_rgb = observation['uri_to_rgb']
        uri_to_annotations = observation['uri_to_annotations']
        uri_to_rivers = observation['uri_to_rivers']

        download_file(uri_to_s2, target_dir + '/s2.tif')
        download_file(uri_to_rgb, target_dir + '/rgb.tif')
        download_file(uri_to_annotations, target_dir + '/annotations.geojson')
        download_file(uri_to_rivers, target_dir + '/rivers.geojson')

        read_tifs_as_pil_image(target_dir)
    return

def download_file(uri, destination):
    response = requests.get(uri, stream=True)
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

def read_tifs_as_pil_image(target_dir):
    # Create an empty list to store the image metadata
    metadata_list = []

    # Search for TIFF image files in the directory
    for image_path in glob.glob(target_dir + "/*.tif"):

        # TODO: Figure out how to download s2 images properly
        if "s2" in image_path:
            continue  

        # Open the image file
        image = Image.open(image_path)

        # Extract basic metadata
        image_size = image.size
        image_height = image.height
        image_width = image.width
        image_format = image.format
        image_mode = image.mode
        image_is_animated = getattr(image, "is_animated", False)
        frames_in_image = getattr(image, "n_frames", 1)

        # Create a dictionary to store the metadata
        metadata = {
            "filename": pathlib.Path(image_path).name,
            "size": image_size,
            "height": image_height,
            "width": image_width,
            "format": image_format,
            "mode": image_mode,
            "is_animated": image_is_animated,
            "frames": frames_in_image,
        }

        # Add the metadata dictionary to the list
        metadata_list.append(metadata)

    # Write the metadata list to a CSV file
    obs_number = target_dir[-1]
    with open(target_dir + "/metadata{}.csv".format(obs_number), "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metadata_list[0].keys())
        writer.writeheader()
        writer.writerows(metadata_list)