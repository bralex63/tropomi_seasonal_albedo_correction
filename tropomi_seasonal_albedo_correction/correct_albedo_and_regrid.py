# The python code in this file was written by Alexander C. Bradley with some being
# adapted from work by Nicholas Balasus for https://doi.org/10.5194/amt-16-3787-2023 

import os
import re
import pickle

# Do not use pygeos for this script, use shapely instead
os.environ["USE_PYGEOS"] = "0"
from netCDF4 import Dataset
import pandas as pd
import geopandas as gpd
import numpy as np
import tensorflow as tf
from shapely import Point, Polygon
from tqdm import tqdm

import settings   

def correct_albedo_and_regrid(
        raw_tropomi_files_path,
        regridded_data_output_directory
        ):
    """
    Load TROPOMI data, correct stripe artifacts, predict values using a model, and regrid the data.
    
    Parameters:
        raw_tropomi_files_path (str): Path to the directory containing raw TROPOMI files. folder
        regridded_data_output_directory (str): Path to the directory where regridded data will be saved. folder
        
    Output:
        Destripes, correct, and regrids each TROPOMI data file and exports it as a csv that contains the following columns:
            'geometry', 'surface_albedo_SWIR', 'surface_pressure', 'surface_altitude', 'xch4_corrected'
    """
    bounds = settings.bounds
    regrid_resolution = settings.resolution
    # Load df and use model to predict new values
    date_orbit_getter = re.compile('S5P_(RPRO|OFFL)_L2__CH4____(\d\d\d\d)(\d\d)(\d\d)T.*_(\d\d\d\d\d)_.*')

    with open(settings.z_stats_path, "r+b") as f:
        z_stats = pickle.load(f)
    tropomi_file_list = os.listdir(raw_tropomi_files_path)
    for file in tqdm(tropomi_file_list):
        
        date_orbit = re.search(date_orbit_getter, file)
        
        date = date_orbit.group(2) + '-' + date_orbit.group(3) + '-' + date_orbit.group(4)
        orbit = date_orbit.group(5)        
        
        df = get_tropomi_df(raw_tropomi_files_path + file)
        df = df[
            (df['latitude']>=bounds[1])&
            (df['latitude']<=bounds[3])&
            (df['longitude']>=bounds[0])&
            (df['longitude']<=bounds[2])]
        df.reset_index(inplace=True)
        
        prediction_df = df.copy()
        if len(prediction_df) == 0:
            continue
        prediction_df = prediction_df[settings.training_data_columns]
        prediction_df = known_z_score(prediction_df, z_stats)
        prediction_array = prediction_df.values
        model = tf.keras.models.load_model(settings.monthly_models_prefix + '_' + date_orbit.group(3))
        y = model.predict(prediction_array)
    
        df['predicted_values'] = y.flatten()
        
        if regrid_resolution is None:
            df['xch4_corrected'] = df['xch4_corrected'] - df['predicted_values']
            df = df[['xch4_corrected', 'scanline', 'ground_pixel']]
            df.to_csv(regridded_data_output_directory + orbit + '_' + date + '.csv', index=False)
            continue

        # Regrid data
        df = df[['latitude_bounds', 'longitude_bounds', 'surface_albedo_SWIR', 'surface_pressure', 'surface_altitude', 'xch4_corrected', 'predicted_values']]
        
        df['geometry'] = [Polygon(((df['longitude_bounds'][x][0], df['latitude_bounds'][x][0]), 
                            (df['longitude_bounds'][x][1], df['latitude_bounds'][x][1]), 
                            (df['longitude_bounds'][x][2], df['latitude_bounds'][x][2]), 
                            (df['longitude_bounds'][x][3], df['latitude_bounds'][x][3])
                            )) for x in range(len(df['longitude_bounds']))]
        geo_df = gpd.GeoDataFrame(df, geometry='geometry')
        
        grid = generate_grid(regrid_resolution, bounds)
        
        regridded = regrid_df(geo_df, grid)
        
        regridded.to_csv(regridded_data_output_directory + orbit + '_' + date + '.csv', index=False)
    

def generate_grid(resolution, bounds):
    """
    Generate a regular grid in a geopandas GeoDataFrame for use in regridding.
    
    Parameters:
        resolution (float): The desired calculation resolution in degrees.
        bounds (list): Latitudinal and longitudinal bounds for regridding calculation.
    
    Returns:
        gdf (geopandas.GeoDataFrame): Regular grid for regridding data.
    """
    # Retrieve bounds data and round to nearest resolution
    xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin, xmax+resolution, resolution)
    y = np.arange(ymin, ymax+resolution, resolution)
    
    # Create each point pair in a shapely geometry
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    geom = [Point(x, y) for x, y in zip(xx, yy)]
    
    # Enter points into GeoDataFrame and make polygons at given resolution
    gdf = gpd.GeoDataFrame(geometry=geom)
    gdf.geometry = gdf.geometry.buffer(resolution/2, cap_style=3)
    
    
    return gdf
    
    
def regrid_df(df, gdf):
    """
   Regrid data to a regular grid using the tesselation oversampling method.
   
   Parameters:
       df (pandas.DataFrame): Dataframe containing the tropomi data.
       resolution (float): The desired calculation resolution in degrees.
       gdf (geopandas.GeoDataFrame): Regular grid for regridding data.
   
   Returns:
       grouped (pandas.DataFrame): Regridded data.
       
    This function regrids according to the tesselation oversampling method as described in sun et. al.
    https://amt.copernicus.org/articles/11/6679/2018/
    """
        
    # number each point for organizational purposes
    gdf['cell'] = [int(x) for x in range(0, len(gdf.index))]
    # Split given dataset onto new grid
    newdf = df.overlay(gdf, how='union')
    newdf = newdf[newdf['cell']>=0]
    
    newdf = newdf[newdf['xch4_corrected']>0]
    # Calculate weighted average and clean data
    newdf['area'] = newdf.geometry.area
    
    tot_areas = newdf.groupby("cell").agg(totarea = pd.NamedAgg(column='area', aggfunc='sum'))
    tot_areas.reset_index(inplace=True)

    newdf = pd.merge(newdf, tot_areas, on='cell')
    newdf['weight'] = newdf['area']/newdf['totarea']
    newdf['xch4_corrected'] = newdf['weight']*newdf['xch4_corrected']
    newdf['predicted_values'] = newdf['weight']*newdf['predicted_values']
    newdf["surface_albedo_SWIR"] = newdf["weight"] * newdf["surface_albedo_SWIR"]
    newdf["surface_pressure"] = newdf["weight"] * newdf["surface_pressure"]
    newdf["surface_altitude"] = newdf["weight"] * newdf["surface_altitude"]
    # newdf = newdf[newdf['methane_mixing_ratio']<10000]
    
    # Group by "cell" and sum the weighted values and fractions
    # Edit method here to add new data to the output
    grouped = newdf.groupby("cell").agg(
        total_xch4_corrected=pd.NamedAgg(column="xch4_corrected", aggfunc="sum"),
        total_predicted_values=pd.NamedAgg(column='predicted_values', aggfunc='sum'),
        total_fraction=pd.NamedAgg(column="weight", aggfunc="sum"),
        total_surface_albedo_SWIR=pd.NamedAgg(column="surface_albedo_SWIR", aggfunc="sum"),
        total_surface_pressure=pd.NamedAgg(column="surface_pressure", aggfunc="sum"),
        total_surface_altitude=pd.NamedAgg(column="surface_altitude", aggfunc="sum")
    )
    
    # Calculate the weighted average
    grouped["xch4_corrected"] = (grouped["total_xch4_corrected"] / grouped["total_fraction"])
    grouped['predicted_values'] = (grouped['total_predicted_values']/grouped['total_fraction'])
    grouped["surface_albedo_SWIR"] = (grouped["total_surface_albedo_SWIR"] / grouped["total_fraction"])
    grouped["surface_pressure"] = (grouped["total_surface_pressure"] / grouped["total_fraction"])
    grouped["surface_altitude"] = (grouped["total_surface_altitude"] / grouped["total_fraction"])

    # Reset the index to have the "cell" as a column
    grouped.reset_index(inplace=True)
    
    # Drop unnecessary columns
    grouped.drop(
        [
            "total_xch4",
            "total_xch4_corrected",
            "total_predicted_values",
            "total_fraction",
            "total_surface_albedo_SWIR",
            "total_surface_pressure",
            "total_surface_altitude",
        ],
        axis=1,
        errors="ignore",
        inplace=True,
    )
    
    grouped = pd.merge(gdf, grouped, on='cell', how='left')
    grouped['xch4_builtin'] = grouped['xch4_corrected']
    grouped['xch4_corrected'] = grouped['xch4_corrected'] - grouped['predicted_values']
    grouped.drop(columns=['cell', 'predicted_values'], inplace=True)
    
    return grouped


def known_z_score(df, z_stats):
    """Calculates the z-score for each column in the dataframe based on the provided statistics.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        z_stats (dict): A dictionary containing the mean and standard deviation for each column.
    
    Returns:
        pd.DataFrame: A dataframe containing the z-score normalized values for each column.
    """
    z_score_df = pd.DataFrame()
    for column in df.columns:
        avg = z_stats[column][0]
        std = z_stats[column][1]
        z_score_df[column] = (df[column] - avg)/std
    return z_score_df


# Function to turn one netCDF TROPOMI file into one pandas dataframe
def get_tropomi_df(tropomi_file):
    """Extracts TROPOMI data from a netCDF file and returns it as a pandas dataframe.
   
   Args:
       tropomi_file (str): Path to the TROPOMI netCDF file.
   
   Returns:
       pd.DataFrame: A dataframe containing the TROPOMI data.
       
   This function written by Nicholas Balasus https://github.com/nicholasbalasus/blended_tropomi_gosat_methane
   and altered for this project
   """
   
    with Dataset(tropomi_file) as ds:
        mask = ds["PRODUCT/qa_value"][:] == 1
        tropomi_df = pd.DataFrame({
                           # non-predictor variables
                           "latitude": ds["PRODUCT/latitude"][:][mask],
                           "longitude": ds["PRODUCT/longitude"][:][mask],
                           "latitude_bounds": list(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"][:][mask]),
                           "longitude_bounds": list(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"][:][mask]),
                           
                           # predictor variables
                           "xch4": ds["PRODUCT/methane_mixing_ratio"][:][mask],
                           "xch4_corrected": ds["PRODUCT/methane_mixing_ratio_bias_corrected"][:][mask],
                           "pressure_interval": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/pressure_interval"][:][mask],
                           "surface_pressure": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure"][:][mask],
                           "solar_zenith_angle": ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"][:][mask],
                           "relative_azimuth_angle": np.abs(180 - np.abs(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_azimuth_angle"][:][mask] -
                                                                         ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_azimuth_angle"][:][mask])),
                           "ground_pixel": np.expand_dims(np.tile(ds["PRODUCT/ground_pixel"][:], (mask.shape[1],1)), axis=0)[mask],
                           "scanline": np.expand_dims(np.tile(ds["PRODUCT/scanline"][:], (mask.shape[2],1)).T, axis=0)[mask],
                           "surface_altitude": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"][:][mask],
                           "surface_altitude_precision": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude_precision"][:][mask],
                           "xch4_apriori": np.sum(ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/methane_profile_apriori"][:][mask]/
                                                  np.expand_dims(np.sum(ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/dry_air_subcolumns"][:][mask], axis=1),axis=1), axis=1)*1e9,
                           "reflectance_cirrus_VIIRS_SWIR": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/reflectance_cirrus_VIIRS_SWIR"][:][mask],
                           "xch4_precision": ds["PRODUCT/methane_mixing_ratio_precision"][:][mask],
                           "fluorescence": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/fluorescence"][:][mask],
                           "co_column": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/carbonmonoxide_total_column"][:][mask],
                           "co_column_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/carbonmonoxide_total_column_precision"][:][mask],
                           "h2o_column": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column"][:][mask],
                           "h2o_column_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column_precision"][:][mask],
                           "aerosol_size": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_size"][:][mask],
                           "aerosol_size_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_size_precision"][:][mask],
                           "aerosol_height": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_mid_altitude"][:][mask],
                           "aerosol_height_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_mid_altitude_precision"][:][mask],
                           "aerosol_column": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_number_column"][:][mask],
                           "aerosol_column_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_number_column_precision"][:][mask],
                           "surface_albedo_SWIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_SWIR"][:][mask],
                           "surface_albedo_SWIR_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_SWIR_precision"][:][mask],
                           "surface_albedo_NIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_NIR"][:][mask],
                           "surface_albedo_NIR_precision": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_NIR_precision"][:][mask],
                           "aerosol_optical_thickness_SWIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_optical_thickness_SWIR"][:][mask],
                           "aerosol_optical_thickness_NIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_optical_thickness_NIR"][:][mask],
                           "chi_square_SWIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/chi_square_SWIR"][:][mask],
                           "chi_square_NIR": ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/chi_square_NIR"][:][mask]#,
                          })
    
    return tropomi_df
