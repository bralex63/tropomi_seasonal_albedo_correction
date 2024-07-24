# -*- coding: utf-8 -*-

# The python code in this file was written by Alexander C. Bradley with some being
# adapted from work by Nicholas Balasus for https://doi.org/10.5194/amt-16-3787-2023 

import os
from datetime import time as ttime
import pickle
import math
import re
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset
import numpy as np
from tqdm import tqdm
from shapely import Point, Polygon
import geopandas as gpd
from tqdm import tqdm
from scipy import interpolate

import settings

def prepare_data_paths():
    """
    Create necessary directories for storing regridded data and model outputs.

    This function ensures that all required directories for storing regridded data,
    model outputs, and figures are created. If the directories already exist, they are
    not recreated.

    The following directories are created based on paths specified in the `settings` module:
        - `regridded_gosat_loc`: Directory for regridded GOSAT data.
        - `regridded_tropomi_loc`: Directory for regridded TROPOMI data.
        - `model_folder_path`: Directory for general model outputs.
        - `monthly_model_folder_path`: Directory for storing monthly model outputs.
        - `regridded_BLND_path`: Directory for regridded BLND data.
        - `fig04_folder_path`: Directory for storing figure 04 related outputs.
        - `regridded_albedo_corrected_data_path`: Directory for regridded albedo-corrected data.
        - `colocated_cropdata_output_path`: Directory for colocated crop data outputs.

    Uses:
        pathlib.Path: To create directories if they do not already exist.

    """
    Path(settings.regridded_gosat_loc).mkdir(parents=True, exist_ok=True)
    Path(settings.regridded_tropomi_loc).mkdir(parents=True, exist_ok=True)
    Path(settings.model_folder_path).mkdir(parents=True, exist_ok=True)
    Path(settings.monthly_model_folder_path).mkdir(parents=True, exist_ok=True)
    Path(settings.regridded_BLND_path).mkdir(parents=True, exist_ok=True)
    Path(settings.fig04_folder_path).mkdir(parents=True, exist_ok=True)
    Path(settings.regridded_albedo_corrected_data_path).mkdir(parents=True, exist_ok=True)
    Path(settings.colocated_cropdata_output_path).mkdir(parents=True, exist_ok=True)


def combine_gosat_tropomi(tropomi_gridded_path, gosat_gridded_path, output_path, balasus=False):
    """
    Combine regridded TROPOMI and GOSAT data based on matching dates and save the result.

    This function merges TROPOMI and GOSAT GeoDataFrames based on matching dates,
    filters the GOSAT data to a specific time range, and performs additional processing
    and cleaning steps. The resulting data is saved to a specified output path as a CSV file.

    Parameters:
    tropomi_gridded_path (str): The path to the directory containing regridded TROPOMI data files.
    gosat_gridded_path (str): The path to the directory containing regridded GOSAT data files.
    output_path (str): The path where the combined and processed data will be saved as a CSV file.
    balasus (bool, optional): Flag to indicate if the Balasus processing step should be applied.
                              Default is False.

    Returns:
    None

    Notes:
    - The function expects the filenames of TROPOMI and GOSAT data files to follow a specific 
      date pattern in the format `YYYY-MM_DD.pkl`.
    - The function filters the GOSAT data to include only records within a specific time range 
      (7 pm to 9 pm UTC).
    - The function merges the TROPOMI and GOSAT GeoDataFrames on the `geometry` column.
    - If `balasus` is False, additional delta calculation and filtering steps are applied 
      based on the `settings.training_data_columns`.
    - If `balasus` is True, the function merges the result with existing training data from 
      `settings.training_data_path`.
    """
    resultdf = None
    tropomi_regridded_list = os.listdir(tropomi_gridded_path)
    gosat_regridded_list = os.listdir(gosat_gridded_path)
    datematch_pattern = re.compile(r'(\d{4})-(\d{2})_(\d{2})\.pkl')
    for file in tqdm(tropomi_regridded_list):
        datematch = re.search(datematch_pattern, file)
        year = datematch.group(1)
        month = datematch.group(2)
        day = datematch.group(3)
        
        gosat_file = [x for x in gosat_regridded_list if str(year)+'-'+str(month)+'-'+str(day) in x]
        
        if len(gosat_file) <1: # Only if there is a matching GOSAT file
            continue
        
        with open(tropomi_gridded_path + file, 'r+b') as f:
            tropomi_gdf = pickle.load(f)
            
        with open(gosat_gridded_path + str(gosat_file[0]), 'r+b') as f:
            gosat_gdf = pickle.load(f)
        
        start_time = ttime(19, 0)  # 7 pm
        end_time = ttime(21, 0)    # 9 pm
        
        # Filter the GeoDataFrame based on the time range
        gosat_gdf = gosat_gdf[(gosat_gdf['time'].dt.time >= start_time) & 
                           (gosat_gdf['time'].dt.time <= end_time)]
        
        if len(gosat_gdf) < 1:
            continue

        gosat_gdf.rename(columns={x:"gosat_"+x for x in gosat_gdf.columns if 'geometry' not in x}, inplace=True)
        
        merged_gdf = pd.merge(tropomi_gdf, gosat_gdf, how='left', on='geometry')
        
        merged_gdf.dropna(subset=['xch4', 'gosat_xch4'], axis=0, inplace=True)
        if len(merged_gdf)<1:
            continue
        
        if resultdf is None:
            resultdf = merged_gdf
        else:
            resultdf = pd.concat([resultdf, merged_gdf])
    resultdf['month'] = resultdf['gosat_time'].dt.month

    if balasus == False:
        delta_tropomi_gosat = []
        for index, row in resultdf.iterrows():
            delta = calculate_delta(row)
            delta_tropomi_gosat.append(delta)
            
        resultdf['delta_tropomi_gosat'] = delta_tropomi_gosat
        keepcols = settings.training_data_columns
        keepcols.extend(['month', 'gosat_xch4', 'delta_tropomi_gosat', 'geometry'])
        resultdf = resultdf[keepcols]
    else:
        df = pd.read_csv(settings.training_data_path)
        df = df[['xch4', 'xch4_corrected', 'surface_albedo_SWIR', 'delta_tropomi_gosat']]
        resultdf = pd.merge(resultdf, df, on=['xch4', 'xch4_corrected'])
        keepcols = ['xch4', 'xch4_corrected', 'xch4_blended', 'surface_albedo_SWIR', 'delta_tropomi_gosat', 'month', 'gosat_xch4']
        resultdf = resultdf[keepcols]

    resultdf.to_csv(output_path, index =False)
   
    
    
def calculate_delta(tropomi_gosat_pair):
    """
    Calculate the difference between TROPOMI and GOSAT XCH4 measurements.

    This function computes the delta (∆) between TROPOMI and GOSAT XCH4 (methane) measurements
    based on the method described in Balasus et al. The calculation involves interpolating 
    the GOSAT apriori methane profile to the TROPOMI pressure grid, applying averaging kernels, 
    and computing the corrected XCH4 values for both TROPOMI and GOSAT.

    Parameters:
    tropomi_gosat_pair (pd.Series or dict): A data structure containing TROPOMI and GOSAT measurement 
                                            data for a single pair. The following keys/columns are expected:
        - 'month'
        - 'gosat_ch4_profile_apriori'
        - 'gosat_xch4_averaging_kernel'
        - 'gosat_pressure_weight'
        - 'gosat_pressure_levels'
        - 'altitude_levels'
        - 'dry_air_subcolumns'
        - 'methane_profile_apriori'
        - 'column_averaging_kernel'
        - 'surface_pressure'
        - 'pressure_interval'
        - 'xch4_corrected'
        - 'gosat_xch4'

    Returns:
    float: The computed delta (∆) between TROPOMI and GOSAT XCH4 measurements in parts per billion (ppb).

    Notes:
    - The function handles the interpolation of GOSAT apriori methane profile to the TROPOMI pressure grid.
    - Adjustments are made to account for a global mean bias of GOSAT relative to GGG2020.
    - The computation includes averaging kernels and pressure levels to align the data from both satellites.

    """
    # This function adapted from Balasus et al. 
    month = tropomi_gosat_pair['month']
    gosat_ch4_profile_apriori = np.array(tropomi_gosat_pair['gosat_ch4_profile_apriori'])
    gosat_xch4_averaging_kernel = np.array(tropomi_gosat_pair['gosat_xch4_averaging_kernel'])
    gosat_pressure_weight = np.array(tropomi_gosat_pair['gosat_pressure_weight'])
    gosat_pressure_levels = np.array(tropomi_gosat_pair['gosat_pressure_levels'])
    altitude_levels = np.array(tropomi_gosat_pair['altitude_levels'])
    dry_air_subcolumns = np.array(tropomi_gosat_pair['dry_air_subcolumns'])
    methane_profile_apriori = np.array(tropomi_gosat_pair['methane_profile_apriori'])
    column_averaging_kernel = np.array(tropomi_gosat_pair['column_averaging_kernel'])

    # Calculate ∆(TROPOMI-GOSAT) using equations (A1-A4)
    gosat_mask = gosat_ch4_profile_apriori[::-1] != -9999.99 # when there are 19 pressure levels #gosat_ch4_profile_apriori
   
    gosat_prior = gosat_ch4_profile_apriori[::-1][gosat_mask] # [ppb]
    gosat_p_levels = 100*gosat_pressure_levels[::-1][gosat_mask] # [Pa]
    f_interp_gosat_prior_to_tropomi_pressure_grid = interpolate.interp1d(gosat_p_levels, gosat_prior, bounds_error=False, fill_value="extrapolate")
    tropomi_pressure_levels = [tropomi_gosat_pair["surface_pressure"] - i*tropomi_gosat_pair["pressure_interval"] for i in np.arange(0,13)][::-1] # [Pa]
    tropomi_pressure_layers = np.array([(tropomi_pressure_levels[i]+tropomi_pressure_levels[i+1])/2 for i in np.arange(0,12)])

    c_Tr = tropomi_gosat_pair["xch4_corrected"] # [ppb]
    h_T = dry_air_subcolumns/np.sum(dry_air_subcolumns) # [unitless]
    A_T = column_averaging_kernel # [unitless]
    x_Ga = f_interp_gosat_prior_to_tropomi_pressure_grid(tropomi_pressure_layers) # [ppb]
    x_Ta = 1e9*methane_profile_apriori/dry_air_subcolumns # [ppb]
    c_Gr = tropomi_gosat_pair["gosat_xch4"] - settings.tccon_offset # [ppb], adjust to GOSAT having a global mean bias of 0 relative to GGG2020
    c_Ga = np.sum(h_T*x_Ga) # [ppb]
    x_Gr = x_Ga * (c_Gr/c_Ga) # [ppb]

    c_T_star = c_Tr + np.sum(h_T*(1-A_T)*(x_Ga-x_Ta)) # [ppb]
    c_G_star = np.sum(h_T*(x_Ga+(A_T*(x_Gr-x_Ga)))) # [ppb]
    delta_tropomi_gosat = c_T_star - c_G_star # [ppb]

    return delta_tropomi_gosat
    

def round_coordinates_in_polygon(geometry, precision=10):
    """
    Round the coordinates of a Polygon to a specified precision.
    
    Parameters:
        geometry (shapely.geometry.Polygon): Polygon geometry.
        precision (int): Number of decimal places to round to.
    
    Returns:
        shapely.geometry.Polygon: Polygon with rounded coordinates.
    """
    # Iterate over the coordinates of each ring in the polygon
    rounded_exterior = [(round(x, precision), round(y, precision)) for x, y in geometry.exterior.coords]
    rounded_interiors = [[(round(x, precision), round(y, precision)) for x, y in ring.coords] for ring in geometry.interiors]
    
    # Create a new polygon with rounded coordinates
    rounded_polygon = Polygon(shell=rounded_exterior, holes=rounded_interiors)
    
    return rounded_polygon


def regrid_tropomi_files(input_loc, res, output_loc, bounds):
    """
    Regrid TROPOMI files to a specified resolution within given geographical bounds.

    This function processes TROPOMI data files, filters them based on geographical bounds,
    and regrids the data to a specified resolution. The regridded data is then saved to 
    the specified output location.

    Parameters:
    input_loc (str): The directory path where the input TROPOMI files are located.
    res (float): The resolution for regridding the data.
    output_loc (str): The directory path where the regridded files will be saved.
    bounds (list): A list specifying the geographical bounds [min_lon, min_lat, max_lon, max_lat].

    Returns:
    None

    Notes:
    - The function supports processing both BLND and non-BLND TROPOMI files.
    - It filters the data based on the specified geographical bounds.
    - Regridding is performed using a specified resolution and the result is saved in .pkl format.
    """
    # Load df and use model to predict new values
    date_orbit_getter = re.compile('S5P_(RPRO|OFFL|BLND)_L2__CH4____(\d\d\d\d)(\d\d)(\d\d)T.*_(\d\d\d\d\d)_.*')
    tropomi_file_list = os.listdir(input_loc)
    for file in tqdm(tropomi_file_list):
        
        date_orbit = re.search(date_orbit_getter, file)
        
        date = date_orbit.group(2) + '-' + date_orbit.group(3) + '_' + date_orbit.group(4)
        orbit = date_orbit.group(5)        
        if date_orbit.group(1) == 'BLND':
            df = get_tropomi_df(input_loc+file, BLND=True)
        else:
            df = get_tropomi_df(input_loc+file)
        df = df[
            (df['latitude']>=bounds[1])&
            (df['latitude']<=bounds[3])&
            (df['longitude']>=bounds[0])&
            (df['longitude']<=bounds[2])]
        if len(df)<1:
            continue
            
        df['geometry'] = [
            Polygon(((row['longitude_bounds'][0], row['latitude_bounds'][0]), 
                     (row['longitude_bounds'][1], row['latitude_bounds'][1]), 
                     (row['longitude_bounds'][2], row['latitude_bounds'][2]), 
                     (row['longitude_bounds'][3], row['latitude_bounds'][3])))
            if len(row['latitude_bounds']) >= 4 and len(row['longitude_bounds']) >= 4
            else None
            for _, row in df.iterrows()
        ]
        
        geo_df = gpd.GeoDataFrame(df, geometry='geometry')
        geo_df.drop(columns=['latitude', 'longitude', 'latitude_bounds', 'longitude_bounds', 
                             'across_track_pixel_index', 
                             'eastward_wind', 'northward_wind',
                             'snow_covered_flag'], errors='ignore', inplace=True)
        
        grid = generate_grid(res, bounds)
        if date_orbit.group(1) == 'BLND':
            regridded = regrid_tropomi_df(geo_df, True, grid, ['xch4',
                                                          'xch4_corrected',
                                                          'xch4_blended'
                                                            ])
        else:
            regridded = regrid_tropomi_df(geo_df, False, grid, ['xch4',
                                                            'xch4_corrected',
                                                            'surface_pressure',
                                                            'pressure_interval',
                                                            'surface_altitude',
                                                            'altitude_levels',
                                                            'surface_altitude_precision',
                                                            'xch4_apriori',
                                                            'reflectance_cirrus_VIIRS_SWIR',
                                                            'xch4_precision',
                                                            'fluorescence',
                                                            'co_column',
                                                            'co_column_precision',
                                                            'h2o_column',
                                                            'h2o_column_precision',
                                                            'aerosol_size',
                                                            'aerosol_size_precision',
                                                            'aerosol_height',
                                                            'aerosol_height_precision',
                                                            'aerosol_column',
                                                            'aerosol_column_precision',
                                                            'surface_albedo_SWIR',
                                                            'surface_albedo_SWIR_precision',
                                                            'surface_albedo_NIR',
                                                            'surface_albedo_NIR_precision',
                                                            'aerosol_optical_thickness_SWIR',
                                                            'aerosol_optical_thickness_NIR',
                                                            'chi_square_SWIR',
                                                            'chi_square_NIR'])
        
        with open(output_loc + orbit + '_' + date + '.pkl', 'w+b') as f:
            pickle.dump(regridded, f)
            

def generate_grid(resolution, bounds):
    """Generates a regular grid in a geopandas GeoDataFrame for use in regridding
        
    Keyword arguments:
        bounds -- lat lon lower and upper limits for regridding calculation in the following list format:
            [longitude minimum, latitude minimum, longitude maximum, latitude maximum]
        resolution -- the desired calculation resolution in degrees

    The grid will start exactly at the specified minimum bounds and will end at the nearest without going over
    the maximum specified bounds multiple of the resolution. To ensure that your regridded data can be properly
    averaged, make sure all of your bound and resolution are teh same for each regridding calculation.
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
    

def regrid_tropomi_df(df, BLND, gdf, columns_to_regrid):
    """
    Regrid the given DataFrame to a new grid with specified columns.

    This function takes a GeoDataFrame and regrids it to a new grid defined by another GeoDataFrame,
    aggregating specified columns using a weighted average based on the area overlap. This uses
    the Tesselation oversampling method as described in Sun et. al. https://amt.copernicus.org/articles/11/6679/2018/

    Parameters:
    df (GeoDataFrame): The input GeoDataFrame containing the data to be regridded.
    BLND (bool): A flag indicating whether the input data is blended (True) or not (False).
    gdf (GeoDataFrame): The GeoDataFrame defining the new grid.
    columns_to_regrid (list of str): List of column names to be regridded.

    Returns:
    GeoDataFrame: The regridded GeoDataFrame with the specified columns aggregated.

    Notes:
    - The function assumes that at least one column exists in columns_to_regrid.
    - It calculates the weighted average of the specified columns based on the area of overlap.
    - Additional columns are included in the output if BLND is False.
    """
    # number each point for organizational purposes
    gdf['cell'] = [int(x) for x in range(0, len(gdf.index))]
    # Split given dataset onto new grid
    newdf = df.overlay(gdf, how='union')
    newdf = newdf[newdf['cell']>=0]
    
    newdf = newdf[newdf[columns_to_regrid[0]] > 0]  # Assuming at least one column exists in columns_to_regrid
    # Calculate weighted average and clean data
    newdf['area'] = newdf.geometry.area
    
    tot_areas = newdf.groupby("cell").agg(totarea = pd.NamedAgg(column='area', aggfunc='sum'))
    tot_areas.reset_index(inplace=True)

    newdf = pd.merge(newdf, tot_areas, on='cell')
    newdf['weight'] = newdf['area']/newdf['totarea']
    
    for col in columns_to_regrid:
        newdf[col] = newdf['weight'] * newdf[col]

    # Group by "cell" and sum the weighted values and fractions
    if BLND:
        grouped = newdf.groupby(["cell"]).agg(
            **{f'total_{col}': (col, 'sum') for col in columns_to_regrid},
            total_fraction=pd.NamedAgg(column="weight", aggfunc="sum")
        )
    else:
        grouped = newdf.groupby(["cell"]).agg(
            **{f'total_{col}': (col, 'sum') for col in columns_to_regrid},
            solar_zenith_angle=('solar_zenith_angle', 'last'),
            relative_azimuth_angle=('relative_azimuth_angle', 'last'),
            surface_classification=('surface_classification', 'last'),
            time=('time', 'last'),
            dry_air_subcolumns =('dry_air_subcolumns', 'last'),
            column_averaging_kernel = ('column_averaging_kernel', 'last'),
            methane_profile_apriori = ('methane_profile_apriori', 'last'),
            ground_pixel = ('ground_pixel', 'last'),
            total_fraction=pd.NamedAgg(column="weight", aggfunc="sum")
        )
    
    # Calculate the weighted average
    for col in columns_to_regrid:
        grouped[col] = grouped[f'total_{col}'] / grouped['total_fraction']
    
    # Reset the index to have the "cell" as a column
    grouped.reset_index(inplace=True)
    
    # Drop unnecessary columns
    for col in columns_to_regrid:
        grouped.drop([f'total_{col}'], axis=1, errors='ignore', inplace=True)
        grouped.drop([f'total_fraction_{col}'], axis=1, errors='ignore', inplace=True)

    grouped = pd.merge(gdf, grouped, on='cell', how='left')
    
    grouped.drop(columns=['cell'], inplace=True)
   
    return grouped


    
def regrid_df(df, resolution, gdf):
    """
    Regrid the given DataFrame to a new grid with specified columns.

    This function takes a GeoDataFrame and regrids it to a new grid defined by another GeoDataFrame,
    aggregating the 'xch4' column using a weighted average based on the area overlap.

    Parameters:
    df (GeoDataFrame): The input GeoDataFrame containing the data to be regridded.
    resolution (float): The resolution of the new grid.
    gdf (GeoDataFrame): The GeoDataFrame defining the new grid.

    Returns:
    GeoDataFrame: The regridded GeoDataFrame with the 'xch4' column aggregated.

    Notes:
    - The function assumes that the 'xch4' column exists in the input DataFrame.
    - It calculates the weighted average of the 'xch4' column based on the area of overlap.
    - Additional columns like 'time' are included in the output with the value from the last overlapping cell.
    """
    # number each point for organizational purposes
    gdf['cell'] = [int(x) for x in range(0, len(gdf.index))]
    # Split given dataset onto new grid
    newdf = df.overlay(gdf, how='union')
    newdf = newdf[newdf['cell']>=0]
    
    newdf = newdf[newdf['xch4']>0]
    # Calculate weighted average and clean data
    newdf['area'] = newdf.geometry.area
    
    tot_areas = newdf.groupby("cell").agg(totarea = pd.NamedAgg(column='area', aggfunc='sum'))
    tot_areas.reset_index(inplace=True)

    newdf = pd.merge(newdf, tot_areas, on='cell')
    newdf['weight'] = newdf['area']/newdf['totarea']
    newdf['xch4'] = newdf['weight']*newdf['xch4']
    
    # Group by "cell" and sum the weighted values and fractions
    # Edit method here to add new data to the output
    grouped = newdf.groupby(["cell"]).agg(
        total_xch4_corrected=('xch4', 'sum'),
        total_fraction=('weight', 'sum'),
        time=('time', 'last')
    )
    
    # Calculate the weighted average
    grouped["xch4"] = grouped["total_xch4_corrected"] / grouped["total_fraction"]
    
    # Reset the index to have the "cell" as a column
    grouped.reset_index(inplace=True)
    
    # Drop unnecessary columns
    grouped.drop(["total_xch4_corrected"], axis=1, errors='ignore', inplace=True)

    grouped = pd.merge(gdf, grouped, on='cell', how='left')
    
    grouped.drop(columns=['cell', 'total_fraction'], inplace=True)
   
    return grouped


def regrid_gosat(df, resolution, gdf):
    """
    Regrid the given GeoDataFrame to a new grid with specified columns.

    This function takes a GeoDataFrame containing GOSAT data and regrids it to a new grid defined 
    by another GeoDataFrame, aggregating various columns using a weighted average based on the 
    area overlap.

    Parameters:
    df (GeoDataFrame): The input GeoDataFrame containing the GOSAT data to be regridded.
    resolution (float): The resolution of the new grid.
    gdf (GeoDataFrame): The GeoDataFrame defining the new grid.

    Returns:
    GeoDataFrame: The regridded GeoDataFrame with the specified columns aggregated.

    Notes:
    - The function assumes that the 'xch4' column exists in the input DataFrame.
    - It calculates the weighted average of the 'xch4' and 'surface_altitude' columns based on the 
      area of overlap.
    - Additional columns like 'time', 'model_xco2_range', 'retr_flag', 'ch4_profile_apriori', 
      'xch4_averaging_kernel', 'pressure_levels', and 'pressure_weight' are included in the output 
      with the value from the last overlapping cell.
    """
    # number each point for organizational purposes
    gdf['cell'] = [int(x) for x in range(0, len(gdf.index))]
    # Split given dataset onto new grid
    newdf = df.overlay(gdf, how='union')
    newdf = newdf[newdf['cell']>=0]
    
    newdf = newdf[newdf['xch4']>0]
    # Calculate weighted average and clean data
    newdf['area'] = newdf.geometry.area
    
    tot_areas = newdf.groupby("cell").agg(totarea = pd.NamedAgg(column='area', aggfunc='sum'))
    tot_areas.reset_index(inplace=True)

    newdf = pd.merge(newdf, tot_areas, on='cell')
    newdf['weight'] = newdf['area']/newdf['totarea']
    newdf['xch4'] = newdf['weight']*newdf['xch4']
    newdf['surface_altitude'] = newdf['weight']*newdf['surface_altitude']
    
    # Group by "cell" and sum the weighted values and fractions
    # Edit method here to add new data to the output
    grouped = newdf.groupby(["cell"]).agg(
        total_xch4_corrected=('xch4', 'sum'),
        total_fraction=('weight', 'sum'),
        time=('time', 'last'),
        model_xco2_range=('model_xco2_range', 'last'),
        surface_altitude = ('surface_altitude', 'sum'),
        retr_flag = ('retr_flag', 'last'),
        ch4_profile_apriori = ('ch4_profile_apriori', 'last'),
        xch4_averaging_kernel = ('xch4_averaging_kernel', 'last'),
        pressure_levels = ('pressure_levels', 'last'),
        pressure_weight = ('pressure_weight', 'last')
        
    )
    
    # Calculate the weighted average
    grouped["xch4"] = grouped["total_xch4_corrected"] / grouped["total_fraction"]
    grouped['surface_altitude'] = grouped['surface_altitude'] / grouped['total_fraction']
    
    # Reset the index to have the "cell" as a column
    grouped.reset_index(inplace=True)
    
    # Drop unnecessary columns
    grouped.drop(["total_xch4_corrected"], axis=1, errors='ignore', inplace=True)

    grouped = pd.merge(gdf, grouped, on='cell', how='left')
    
    grouped.drop(columns=['cell', 'total_fraction'], inplace=True)
   
    return grouped


# Function to turn a list of netCDF GOSAT files into one pandas dataframe
def regrid_gosat_files(gosat_file_location, res, output_loc, bounds):
    """
    Regrid GOSAT files to a specified resolution and save the results.

    This function reads GOSAT files from the given location, applies quality masking, 
    filters data within the specified bounds, and regrids the data to the desired 
    resolution. The regridded data is saved to the specified output location.

    Parameters:
    gosat_file_location (str): Path to the directory containing GOSAT files.
    res (float): The resolution of the new grid.
    output_loc (str): Path to the directory where the regridded files will be saved.
    bounds (tuple): A tuple specifying the bounds (min_longitude, min_latitude, 
                    max_longitude, max_latitude) within which to filter the data.

    Notes:
    - The function reads GOSAT files, applies a mask based on the quality flag, 
      and filters data within the specified latitude and longitude bounds.
    - It converts the data to a GeoDataFrame, calculates buffer diameters based on 
      latitude, and creates circular buffers around each point.
    - The data is then regridded using the `regrid_gosat` function and saved as a 
      pickle file in the output location.
    """
    gosat_files = os.listdir(gosat_file_location)
    base_grid = generate_grid(res, bounds)
    # gosat_dfs = []
    for gosat_file in tqdm(gosat_files):
        gosat_file = gosat_file_location + gosat_file
        with Dataset(gosat_file) as ds:
            mask = ds["xch4_quality_flag"][:] == 0
            df = pd.DataFrame({
                "latitude": ds["latitude"][:][mask],
                "longitude": ds["longitude"][:][mask],
                "xch4": ds["xch4"][:][mask],
                "model_xco2_range": ds["model_xco2_range"][:][mask],
                "surface_altitude": ds["surface_altitude"][:][mask],
                "time": ds["time"][:][mask],
                "retr_flag": ds["retr_flag"][:][mask],
                "ch4_profile_apriori": list(ds["ch4_profile_apriori"][:][mask]),
                "xch4_averaging_kernel": list(ds["xch4_averaging_kernel"][:][mask]),
                "pressure_levels": list(ds["pressure_levels"][:][mask]),
                "pressure_weight": list(ds["pressure_weight"][:][mask])
            })

        df["time"] = pd.to_datetime(df["time"], unit="s")
        date = df['time'].dt.date[0]
        # gosat_dfs.append(df)
        df = df[(df['longitude']>= bounds[0]) & (df['longitude']<=bounds[2])]
        df = df[(df['latitude']>= bounds[1]) & (df['latitude']<=bounds[3])]
        if len(df) < 1:
            continue
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
        # Apply the function to calculate buffer diameters for each point
        gdf['buffer_diameter'] = gdf['geometry'].apply(lambda point: distance_at_latitude(point.y, 10))
        
        # Create circular buffers around each point
        gdf.geometry = gdf.apply(lambda row: row['geometry'].buffer(row['buffer_diameter'] / 2), axis=1)
        grid = base_grid.copy()
        gridded_gdf = regrid_gosat(gdf, 0.1, grid)
        # gridded_gdf['date'] = date]
        # gridded_gdf.to_csv('test.csv')
        # return
        with open(output_loc+'GOSAT_regridded_'+str(res).replace('.', 'p')+'_' + str(date) + '.pkl', 'w+b') as f:
            pickle.dump(gridded_gdf, f)


def distance_at_latitude(lat, dis):
    """
    Calculate the angular distance in degrees of longitude corresponding to a given linear distance at a specific latitude.

    This function computes how many degrees of longitude correspond to a linear distance at a given latitude, taking into account the 
    Earth's curvature. This is useful for determining the spatial extent of circular buffers or other geographical calculations.

    Parameters:
    lat (float): The latitude at which the calculation is to be performed, in degrees.
    dis (float): The linear distance for which the corresponding angular distance is to be calculated, in meters.

    Returns:
    float: The angular distance in degrees of longitude that corresponds to the given linear distance at the specified latitude.

    Notes:
    - The length of one degree of longitude varies with latitude due to the Earth's spherical shape.
    - The result is an approximation assuming a spherical Earth.
    """
    # Calculate the length of one degree of longitude at the given latitude
    length = 111.32 * math.cos(math.radians(lat))
    
    # Calculate the distance in degrees for longitude
    dis = dis / (length * math.cos(math.radians(lat)))
    # print("Distance in decimal degrees:", dis)
    return dis
    
   
# Function to turn one netCDF TROPOMI file into one pandas dataframe
def get_tropomi_df(tropomi_file, BLND=False):
    """
    Extract data from a TROPOMI file and return it as a pandas DataFrame.

    This function reads TROPOMI data from a NetCDF file, processes the data based on the given format (BLND or non-BLND), 
    and returns it as a pandas DataFrame. The processing includes applying masks, converting arrays to lists, 
    and handling datetime conversions.

    Parameters:
    tropomi_file (str): The path to the TROPOMI NetCDF file.
    BLND (bool): If True, processes data in blended format. If False, processes data in non-blended format.

    Returns:
    pd.DataFrame: A DataFrame containing the processed TROPOMI data. The DataFrame includes variables such as latitude, longitude, 
                  methane mixing ratio, and various supporting data, depending on the format specified.

    Notes:
    - For non-BLND files, the function processes a comprehensive set of variables related to methane concentration and supporting data.
    - For BLND files, the function processes a more limited set of variables focused on methane mixing ratios.
    - The 'time' column is converted to a pandas datetime format for both types of files.
    """
    # This code adapted from Balasus et al.
    if not BLND:
        with Dataset(tropomi_file) as ds:
            mask = ds["PRODUCT/qa_value"][:] == 1
            tropomi_df = pd.DataFrame({
                            # non-predictor variables
                            "latitude": ds["PRODUCT/latitude"][:][mask],
                            "longitude": ds["PRODUCT/longitude"][:][mask],
                            "time": np.expand_dims(np.tile(ds["PRODUCT/time_utc"][:][0,:], (mask.shape[2],1)).T, axis=0)[mask],
                            "latitude_bounds": list(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"][:][mask]),
                            "longitude_bounds": list(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"][:][mask]),
                            "xch4": ds["PRODUCT/methane_mixing_ratio"][:][mask],
                            "xch4_corrected": ds["PRODUCT/methane_mixing_ratio_bias_corrected"][:][mask],
                            "pressure_interval": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/pressure_interval"][:][mask],
                            "surface_pressure": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure"][:][mask],
                            "dry_air_subcolumns": list(ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/dry_air_subcolumns"][:][mask]),
                            "methane_profile_apriori": list(ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/methane_profile_apriori"][:][mask]),
                            "column_averaging_kernel": list(ds["PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/column_averaging_kernel"][:][mask]),
                            "altitude_levels": list(ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/altitude_levels"][:][mask]),
                            # predictor variables
                            "solar_zenith_angle": ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"][:][mask],
                            "relative_azimuth_angle": np.abs(180 - np.abs(ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_azimuth_angle"][:][mask] -
                                                                            ds["PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_azimuth_angle"][:][mask])),
                            "ground_pixel": np.expand_dims(np.tile(ds["PRODUCT/ground_pixel"][:], (mask.shape[1],1)), axis=0)[mask],
                            "surface_classification": (ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_classification"][:][mask] & 0x03).astype(int),
                            "surface_altitude": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"][:][mask],
                            "surface_altitude_precision": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude_precision"][:][mask],
                            "eastward_wind": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind"][:][mask],
                            "northward_wind": ds["PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind"][:][mask],
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

        # Convert column of strings to datetime
        tropomi_df["time"] = pd.to_datetime(tropomi_df["time"], format="%Y-%m-%dT%H:%M:%S.%fZ")
        return tropomi_df
    else:
        with Dataset(tropomi_file) as ds:
            mask = ds["qa_value"][:] == 1
            tropomi_df = pd.DataFrame({
                            # non-predictor variables
                            "latitude": ds["latitude"][:][mask],
                            "longitude": ds["longitude"][:][mask],
                            "latitude_bounds": list(ds["latitude_bounds"][:][mask]),
                            "longitude_bounds": list(ds["longitude_bounds"][:][mask]),
                            "xch4": ds["methane_mixing_ratio"][:][mask],
                            "xch4_corrected": ds["methane_mixing_ratio_bias_corrected"][:][mask],
                            "xch4_blended": ds["methane_mixing_ratio_blended"][:][mask]
                            })

            # Convert column of strings to datetime
            return tropomi_df