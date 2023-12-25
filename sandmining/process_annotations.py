import geojson 
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import profiles
from rasterio.features import rasterize
from shapely.geometry import mapping, Point, Polygon
from shapely.geometry import shape
from shapely.ops import unary_union

from project_config import DATA_DIRECTORY

import numpy as np
import os

def process_annotations():
    """
    For each of the observations in the data directory, get the height and the width
    as well as the file path for the .tif RGB image. 
        Step 1: Rasterize the annotations and rivers.
        Step 2: Save them the data directory.
        Step 3: Visualize the rasterized shapes on top of the RGB image. 
    """
    for observation_name in os.listdir(DATA_DIRECTORY):
        if (not ".DS_Store" in observation_name) and (os.path.isdir(DATA_DIRECTORY / observation_name)):
            OBSERVATION_DIRECTORY = DATA_DIRECTORY / '{}'.format(observation_name)
            obs_number = observation_name[-1]
            METADATA_DIRECTORY = OBSERVATION_DIRECTORY / 'metadata{}.csv'.format(obs_number)
            metadata = pd.read_csv(METADATA_DIRECTORY)
            height = metadata['height'].values[0]
            width = metadata['width'].values[0]
            print(obs_number, height, width)
            LABELS_GEOJSON = OBSERVATION_DIRECTORY / 'annotations.geojson'
            RIVERS_GEOJSON = OBSERVATION_DIRECTORY / 'rivers.geojson' 
            generate_mask(raster_path=OBSERVATION_DIRECTORY / 'rgb.tif', shape_path=LABELS_GEOJSON, 
                          output_path=OBSERVATION_DIRECTORY, file_name='labels_mask_obs{}.tif'.format(obs_number))
            generate_mask(raster_path=OBSERVATION_DIRECTORY / 'rgb.tif', shape_path=RIVERS_GEOJSON, 
                          output_path=OBSERVATION_DIRECTORY, file_name='rivers_mask_obs{}.tif'.format(obs_number))
    return

def generate_mask(raster_path, shape_path, output_path, file_name):
    """Function that generates a binary mask from a vector file (shp or geojson)
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.
    """
    #load raster
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
    
    #load o shapefile ou GeoJson
    with open(shape_path, "r", encoding="utf-8") as f:
        geojson_features = geojson.load(f)
    polys = geojson_features['features']
    
    polys = []
    for thing in geojson_features['features']:
        # Extract the geometry coordinates and type
        geometry_data = thing['geometry']
        coordinates = geometry_data["coordinates"][0]
        geometry_type = geometry_data["type"]
        
        # Check if the geometry type is supported
        if geometry_type != "Polygon":
            raise ValueError("Only Polygon geometry type is currently supported.")
        
        # Create the Shapely Polygon object
        polygon = Polygon(coordinates)

        # Append to list
        polys.append(polygon)

        # Access additional properties if needed
        if 'Confidence' in thing['properties'].keys():
            confidence = thing['properties']['Confidence']

    train_df = gpd.GeoDataFrame({'geometry' : polys})
    
    train_df = train_df.set_crs(epsg=4326)
    train_df = train_df.to_crs(epsg=4326)

    #Verify crs
    if train_df.crs != src.crs:
        print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,train_df.crs))
        
        
    #Function that generates the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = unary_union(polygon)
        for i in np.array(poly.exterior.coords):

            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly
    
    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size,
                     all_touched=True)
    
    #Salve
    mask = mask.astype("uint8")
    
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    os.chdir(output_path)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)