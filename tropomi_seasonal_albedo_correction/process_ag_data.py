# -*- coding: utf-8 -*-
# The python code in this file was written by Alexander C. Bradley

import pickle
import os

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.enums import Resampling

import settings

    
def process_ag_data(ag_raster_path, downsampled_raster_output_path, output_pickle_path):
    """
    Downsamples a raster TIFF file, converts it to a GeoDataFrame, and saves it to a pickle file.

    This function performs the following steps:
    1. Opens the input TIFF file and reads the image data.
    2. Downsamples the image data based on a specified downsampling factor.
    3. Updates the raster metadata to reflect the new resolution and saves the downsampled image to a new TIFF file.
    4. Converts the pixel coordinates of the downsampled raster image to geographic coordinates.
    5. Creates a Pandas DataFrame from the geographic coordinates and pixel values.
    6. Converts the DataFrame to a GeoDataFrame.
    7. Sets the coordinate reference system (CRS) for the GeoDataFrame if available.
    8. Saves the GeoDataFrame to a pickle file for future use.

    Args:
        ag_raster_path (str): Path to the input TIFF file containing the raster data.
        downsampled_raster_output_path (str): Path to save the downsampled TIFF file.
        output_pickle_path (str): Path to save the resulting GeoDataFrame as a pickle file.

    Returns:
        None: This function performs file I/O operations and does not return any value.

    Notes:
        - The downsampling factor should be defined in the `settings` module.
        - The GeoDataFrame is created with columns for geographic coordinates (`x`, `y`) and raster values (`value`).
        - Ensure that `settings` is properly configured with the `downsampling_factor`.
    """
    # Open the TIFF file
    with rasterio.open(ag_raster_path) as dataset:
        # Read the image data
        data = dataset.read(1, out_shape=(dataset.height // settings.downsampling_factor, dataset.width // settings.downsampling_factor), resampling=Resampling.mode)
    
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        # Create a new profile to save the downsampled image
        profile = dataset.profile
        profile.update(
            transform=transform,
            width=data.shape[-1],
            height=data.shape[-2]
        )
    
        # Save the downsampled image
        with rasterio.open(downsampled_raster_output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    
    # Open the TIFF file
    with rasterio.open(downsampled_raster_output_path) as src:
        # Read the image data
        data = src.read(1)
    
        # Get the transform to convert pixel coordinates to geographic coordinates
        transform = src.transform
    
    # Convert pixel coordinates to geographic coordinates
    # You need to create a grid of coordinates corresponding to each pixel
    # Here's an example to create such a grid
    rows, cols = data.shape
    x = np.arange(0, cols) * transform.a + transform.c
    y = np.arange(0, rows) * transform.e + transform.f
    x_coords, y_coords = np.meshgrid(x, y)
    
    # Reshape the coordinates and values into a Pandas DataFrame
    df = pd.DataFrame({
        'x': x_coords.flatten(),
        'y': y_coords.flatten(),
        'value': data.flatten()
    })
    
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    
    # Set the coordinate reference system (CRS) if available in the TIFF file
    if src.crs:
        gdf.crs = src.crs
    
    # Now you have a GeoDataFrame with the points loaded from the TIFF file
    with open(output_pickle_path, 'w+b') as f:
        pickle.dump(gdf, f)



def correlate_corrected_data_with_agriculture(regridded_files_path, downsampled_cropdata_pickle_path, colocated_cropdata_output_path):
    """
    Correlates corrected data with agricultural data by regridding and averaging multiple GeoDataFrames and 
    performing spatial joins.

    This function performs the following steps for each month from January to December:
    1. Loads and processes regridded files containing corrected data for the month.
    2. Converts each file into a GeoDataFrame and calculates the predicted values.
    3. Averages the GeoDataFrames to obtain a single GeoDataFrame per month.
    4. Loads the downsampled crop data from a pickle file and prepares it for spatial joining.
    5. Performs a spatial join between the averaged corrected data and the crop data.
    6. Saves the resulting correlated data to a CSV file.

    Args:
        regridded_files_path (str): Path to the directory containing the regridded corrected data files.
        downsampled_cropdata_pickle_path (str): Path to the pickle file containing the downsampled crop data as a GeoDataFrame.
        colocated_cropdata_output_path (str): Directory path where the resulting CSV files with correlated data will be saved.

    Returns:
        None: This function performs file I/O operations and does not return any value.

    Notes:
        - Ensure that the regridded files follow a naming convention that includes the month.
        - The CRS for the GeoDataFrames is set to EPSG:4326; adjust if needed.
        - The `predicted_values` are calculated as the negative difference between 'xch4_builtin' and 'xch4_corrected'.
        - The spatial join uses the 'within' predicate to combine the crop data with the corrected data.
    """
    def load_geodataframes(files_list, file_path):
        geodataframes = []
        for filename in files_list:
            df = pd.read_csv(file_path + filename)
            df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
            df = gpd.GeoDataFrame(df, geometry='geometry')
            df['predicted_values'] = -(df['xch4_builtin'] - df['xch4_corrected'])
            #with open(file_path + filename, 'rb') as f:
             #   geodataframe = pickle.load(f)
            geodataframes.append(df)
        return geodataframes
    
    def average_geodataframes(geodataframes):
      avg_gdf = gpd.GeoDataFrame()
      for col in geodataframes[0].columns:
          if col != 'geometry':
              avg_values = [gdf[col] for gdf in geodataframes]
              avg_values = [np.nanmean(values) for values in zip(*avg_values)]
              avg_gdf[col] = avg_values
      avg_gdf['geometry'] = geodataframes[0].geometry
   
      return avg_gdf
            
    for month in range(1, 13):
        all_files = os.listdir(regridded_files_path)
        month_files = [x for x in all_files if '-'+str(month).zfill(2)+'-' in x]
        geodataframes = load_geodataframes(month_files, regridded_files_path)
        # Calculate the average for the month
        average_gdf = average_geodataframes(geodataframes)
        average_gdf.set_crs("epsg:4326", inplace=True)

        with open(downsampled_cropdata_pickle_path, 'r+b') as f:
            cropdata = pickle.load(f)
            cropdata.set_geometry('geometry', inplace=True)

        # Generate monthly dataframes with cropdata
        combined_cropdata_gdf = gpd.sjoin(cropdata, average_gdf, how="inner", predicate='within')
        combined_cropdata_gdf.dropna(subset=['xch4_corrected'], inplace=True)
        combined_cropdata_gdf.to_csv(colocated_cropdata_output_path + 'colocated_cropdata_'+str(month).zfill(2)+'.csv')