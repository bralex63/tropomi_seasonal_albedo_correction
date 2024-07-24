# -*- coding: utf-8 -*-
# The python code in this file was written by Alexander C. Bradley

# Custom modules
from train_models_4 import load_and_preprocess_data
import settings

# Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, sem, t, ttest_ind
import tensorflow as tf
import shap

# Standard library
import pickle
import os



    
def fig02(paired_data_path, output_path):
    """
    Generates a series of heatmaps to visualize the relationship between surface albedo and the ratio of GOSAT to corrected XCH4 measurements, 
    categorized by overall dataset, summer months, and winter months. The heatmaps display the joint distribution of these variables.

    This function performs the following steps:
    1. **Data Wrangling**:
       - Loads and preprocesses data from a CSV file.
       - Computes a ratio between GOSAT XCH4 and corrected XCH4 values.
       - Filters out outliers and drops NaN values.
       - Separates the data into summer and winter subsets.
       - Calculates Pearson correlation coefficients for the overall, summer, and winter datasets.

    2. **Plotting**:
       - Creates a figure with three subplots, each representing the overall dataset, summer data, and winter data.
       - Generates heatmaps for each dataset using 2D histograms.
       - Adds a reference line and colorbars to the plots.
       - Displays Pearson correlation coefficients on each subplot.

    Args:
        paired_data_path (str): Path to the CSV file containing the paired data with columns such as 'surface_albedo_SWIR', 'gosat_xch4', 'xch4_corrected', and 'month'.
        output_path (str): Path to save the resulting figure as a PNG file.

    Returns:
        None: The function saves the figure to the specified path and does not return any value.

    Notes:
        - The function assumes that the CSV file has columns 'surface_albedo_SWIR', 'gosat_xch4', 'xch4_corrected', and 'month'.
        - The heatmaps are generated with the `Greys` colormap and the Pearson R values are displayed above each subplot.
        - The x-axis and y-axis bins for the heatmaps are defined with specific ranges and number of bins, which might need adjustment based on the data range.
    """
    # Data Wrangling
    # Yearly data
    all_data_df = pd.read_csv(paired_data_path)
    
    #all_data_df['ratio'] = (all_data_df['xch4_corrected'] - all_data_df['delta_tropomi_gosat']) / all_data_df['xch4']
    all_data_df['ratio'] = all_data_df['gosat_xch4'] / all_data_df['xch4_corrected']
    all_data_df = all_data_df[all_data_df['ratio']<1.05]
    all_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data_df = all_data_df.dropna()
    # Seasonal data - summer
    summer_months = [6, 7, 8]
    winter_months = [1, 2, 3]
    summer_df = all_data_df[all_data_df['month'].isin(summer_months)]
    winter_df = all_data_df[all_data_df['month'].isin(winter_months)]

    all_data_pearson = pearsonr(all_data_df['surface_albedo_SWIR'], all_data_df['ratio'])
    summer_pearson = pearsonr(summer_df['surface_albedo_SWIR'], summer_df['ratio'])
    winter_pearson = pearsonr(winter_df['surface_albedo_SWIR'], winter_df['ratio'])

    # Plotting
    # Define grid parameters (adjust these according to your data range)
    x_bins = np.linspace(0, 0.4, 60)  # Number of bins on x-axis
    y_bins = np.linspace(0.9, 1.1, 60)  # Number of bins on y-axis
    
    # Compute histograms for both datasets
    hist1, xedges1, yedges1 = np.histogram2d(all_data_df['surface_albedo_SWIR'], all_data_df['ratio'], bins=(x_bins, y_bins))
    hist2, xedges2, yedges2 = np.histogram2d(summer_df['surface_albedo_SWIR'], summer_df['ratio'], bins=(x_bins, y_bins))
    hist3, xedges3, yedges3 = np.histogram2d(winter_df['surface_albedo_SWIR'], winter_df['ratio'], bins=(x_bins, y_bins))

    straight_x = [0, 0.4]
    straight_y = [1, 1]
    # Create the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(5, 7))
    
    # Plot heatmap for dataset 1
    im1 = axs[0].imshow(hist1.T, extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], origin='lower', cmap='Greys')
    axs[0].plot(straight_x, straight_y, color='k', linewidth=1, linestyle='--')
    axs[0].set_title('April 2018 - December 2022')
    
    # Plot heatmap for dataset 2
    im2 = axs[1].imshow(hist2.T, extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], origin='lower', cmap='Greys')
    axs[1].plot(straight_x, straight_y, color='k', linewidth=1, linestyle='--')
    axs[1].set_title('Summers')
    
    # Plot heatmap for dataset 3
    im3 = axs[2].imshow(hist3.T, extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]], origin='lower', cmap='Greys')
    axs[2].plot(straight_x, straight_y, color='k', linewidth=1, linestyle='--')
    axs[2].set_title('Winters')
    
    # Set a single label for x and y axes for the entire figure
    fig.text(0.5, 0, 'Surface Albedo SWIR', ha='center')
    fig.text(0.0, 0.5, 'GOSAT / TROPOMI XCH4', va='center', rotation='vertical')
    
    # Add colorbars for each subplot
    cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.019)
    cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.019)
    cbar3 = fig.colorbar(im3, ax=axs[2], fraction=0.019)
    
    cbar1.set_label('Counts', rotation=270, labelpad=16)
    cbar2.set_label('Counts', rotation=270, labelpad=16)
    cbar3.set_label('Counts', rotation=270, labelpad=16)
    
    # Pearson coefficients
    axs[0].text(0.22, 1.08, 'Pearson R: ' + str(round(all_data_pearson[0], 3)))
    axs[1].text(0.22, 1.08, 'Pearson R: ' + str(round(summer_pearson[0], 3)))
    axs[2].text(0.22, 1.08, 'Pearson R: ' + str(round(winter_pearson[0], 3)))
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=1200)


def fig03(data_output_path):
    """
    Computes and saves the Pearson correlation coefficients between surface albedo and the ratio of GOSAT to CH4 measurements for different models on a monthly basis.

    This function performs the following steps:
    1. **Calculates Pearson Correlation**:
       - For each model type ('tensorflow', 'uncorrected', 'built_in', 'harvard'), it computes the Pearson correlation coefficient between surface albedo and the ratio of GOSAT to CH4 measurements for each month from January to December.
       - The correlation is calculated based on the specific predictions or data corresponding to each model type.

    2. **Data Retrieval**:
       - For 'tensorflow' models, it loads a pre-trained TensorFlow model and makes predictions to compute the required values.
       - For 'uncorrected' and 'built_in' models, it uses the raw or corrected data from a CSV file.
       - For 'harvard' models, it retrieves and processes data from a combined dataset.

    3. **Saves Results**:
       - Stores the computed Pearson correlation coefficients in a CSV file with the specified output path.

    Args:
        data_output_path (str): Path to save the resulting CSV file containing the Pearson correlation coefficients for each model type and month.

    Returns:
        None: The function saves the Pearson correlation coefficients to the specified CSV file and does not return any value.

    Notes:
        - The function relies on precomputed z-scores for standardization, loaded from a pickle file.
        - The monthly Pearson values are calculated for model types: 'tensorflow', 'uncorrected', 'built_in', and 'harvard'.
        - The function assumes that model files, training data files, and paths are correctly set in the `settings` module.
    """
    def monthly_pearson_value(modeltype):
        pearsonlist = []
        with open(settings.z_stats_path, 'r+b') as f:
            z_stats = pickle.load(f)
        for month in range(1, 13):
            
            if modeltype == 'tensorflow':
                X_train, y_train, X_test, y_test, features = load_and_preprocess_data(settings.training_fraction, month)
                model = tf.keras.models.load_model(settings.monthly_models_prefix + '_' + str(month).zfill(2)) 
                y_pred = model.predict(X_test)
                xch4 = (X_test[:, 4]*z_stats['xch4_corrected'][1]) + z_stats['xch4_corrected'][0]
                gosat_calc = xch4 - y_test
                pred_values = xch4 - y_pred.flatten()
                surface_albedo = (X_test[:, -8]*z_stats['surface_albedo_SWIR'][1]) + z_stats['surface_albedo_SWIR'][0]
                pearson, _ = pearsonr(surface_albedo, (gosat_calc / pred_values))

            elif modeltype == 'uncorrected':
                df = pd.read_csv(settings.training_data_path, encoding='latin-1')
                df = df[df['month']==month]
                pearson, _ = pearsonr(df['surface_albedo_SWIR'], (df['gosat_xch4'] / df['xch4']))

            elif modeltype == 'built_in':
                df = pd.read_csv(settings.training_data_path, encoding='latin-1')
                df = df[df['month']==month]
                pearson, _ = pearsonr(df['surface_albedo_SWIR'], (df['gosat_xch4'] / df['xch4_corrected']))
                
            elif modeltype == 'harvard':
                df = pd.read_csv(settings.combined_BLND_gosat_path)
                df = df[df['month']==month]
                df['gosat_calc'] = df['xch4_corrected'] - df['delta_tropomi_gosat']
                pearson, _ = pearsonr(df['surface_albedo_SWIR'], (df['gosat_calc'] / df['xch4_corrected']))

            pearsonlist.append(pearson)

        return pearsonlist
    
    # Initialize a dictionary to store Pearson values for each model
    pearson_dict = {}
    
    # Iterate over each model
    for i, model in enumerate(['uncorrected', 'built_in', 'harvard', 'tensorflow']):
        pearsonlist = monthly_pearson_value(model)
        pearson_dict[model] = pearsonlist
    pearson_df = pd.DataFrame.from_dict(pearson_dict)
    pearson_df.to_csv(data_output_path)


def fig04(decision_plot_data_prefix, cwave_wavewave_prefix, shap_values_pickle_path=None, expected_values_pickle_path=None, overwrite=False):
    """
    Computes SHAP (SHapley Additive exPlanations) values for monthly models and generates decision plot data and color scale files.

    This function performs the following steps:
    1. **Model Loading and Data Preparation**:
       - Loads monthly models (TensorFlow) and corresponding training data for each month from January to December.
       - Prepares data for SHAP value calculation.

    2. **SHAP Value Calculation**:
       - Calculates SHAP values and expected values for each monthly model using the `shap.SamplingExplainer`.
       - Saves the SHAP values and expected values to pickle files if specified or if they do not already exist.

    3. **Data Storage and Processing**:
       - Loads or re-loads SHAP values and expected values from pickle files if not overwriting.
       - Saves SHAP decision plot data into CSV files with accumulated SHAP values for each month.
       - Computes and saves color wave values based on the `xch4_apriori` values from the SHAP decision plot data.

    Args:
        decision_plot_data_prefix (str): Prefix for the output CSV files containing SHAP decision plot data.
        cwave_wavewave_prefix (str): Prefix for the output CSV files containing color wave and wavewave data.
        shap_values_pickle_path (str, optional): Path to the pickle file for saving or loading SHAP values. Default is None.
        expected_values_pickle_path (str, optional): Path to the pickle file for saving or loading expected values. Default is None.
        overwrite (bool, optional): If True, will overwrite existing SHAP values and expected values files. Default is False.

    Returns:
        None: The function saves SHAP decision plot data and color wave data to CSV files and does not return any value.

    Notes:
        - The function assumes that model files and training data are correctly set in the `settings` module.
        - SHAP values are calculated using the `shap.SamplingExplainer` method.
        - Color wave and wavewave values are computed based on the normalized `xch4_apriori` values.
    """
    if ((overwrite == True) or (shap_values_pickle_path is None and expected_values_pickle_path is None) or
         not (os.path.isfile(shap_values_pickle_path) and os.path.isfile(expected_values_pickle_path))):
        monthly_models = []
        monthly_data = []

        for month in range(1, 13):
            model = tf.keras.models.load_model(settings.monthly_models_prefix + '_' + str(month).zfill(2))
            X_train, y_train, X_test, y_test, features = load_and_preprocess_data(settings.training_fraction, month)
            X_train_np = np.array(X_train)
            monthly_models.append(model)
            monthly_data.append(X_train_np)

        # Create lists to store SHAP values and other plotting information for each month
        shap_values_list = []
        expected_values_list = []

        # Calculate SHAP values for each monthly model
        for i, model in enumerate(monthly_models):
            explainer = shap.SamplingExplainer(model, monthly_data[i])
            shap_values = explainer.shap_values(monthly_data[i])
            shap_values_list.append(shap_values)
            expected_values_list.append(explainer.expected_value)
            print('Finished month', i)

    if (((overwrite == True) and (shap_values_pickle_path is not None and expected_values_pickle_path is not None)) or
        not (os.path.isfile(shap_values_pickle_path) and os.path.isfile(expected_values_pickle_path))):
        # Save SHAP values and other plotting information into pickle files
        with open(shap_values_pickle_path, 'wb') as f:
            pickle.dump(shap_values_list, f)
            
        with open(expected_values_pickle_path, 'wb') as f:
            pickle.dump(expected_values_list, f)

    if (overwrite == False) and (shap_values_pickle_path is not None and expected_values_pickle_path is not None):
        # Reload shap and expected values
        with open(shap_values_pickle_path, 'rb') as f:
            shap_values_list = pickle.load(f)
            
        with open(expected_values_pickle_path, 'rb') as f:
            expected_values_list = pickle.load(f)
    
    decision_plot_order = ['xch4_apriori',
                            'surface_albedo_SWIR_precision',
                            'xch4_corrected',
                            'chi_square_SWIR',
                            'xch4',
                            'surface_albedo_NIR',
                            'aerosol_column',
                            'reflectance_cirrus_VIIRS_SWIR',
                            'surface_pressure',
                            'pressure_interval',
                            'chi_square_NIR',
                            'surface_albedo_SWIR',
                            'co_column_precision',
                            'h2o_column_precision',
                            'xch4_precision',
                            'ground_pixel',
                            'aerosol_column_precision',
                            'relative_azimuth_angle',
                            'aerosol_size_precision',
                            'fluorescence',
                            'h2o_column',
                            'aerosol_optical_thickness_NIR',
                            'co_column',
                            'aerosol_size',
                            'surface_altitude',
                            'surface_albedo_NIR_precision',
                            'aerosol_height_precision',
                            'solar_zenith_angle',
                            'surface_altitude_precision',
                            'aerosol_optical_thickness_SWIR',
                            'aerosol_height',
                            'expected_value'
                            ]

    # Save SHAP decision plot data into separate CSV files
    for i, (shap_values, expected_value) in enumerate(zip(shap_values_list, expected_values_list)):
        columns = settings.training_data_columns
        columns.append('expected_value')
        shap_values_df = pd.DataFrame(shap_values, columns=columns)
        shap_values_df['expected_value'] = expected_value[0]
        shap_values_df = shap_values_df[decision_plot_order]
        #shap_values_df = shap_values_df[decision_plot_order]
        shap_values_df = shap_values_df.transpose()
        for j in range(len(shap_values_df) - 2, -1, -1):
            shap_values_df.iloc[j] = shap_values_df.iloc[j] + shap_values_df.iloc[j + 1]
        shap_values_df.to_csv(decision_plot_data_prefix + f'shap_decision_plot_month_{i+1}_trans.csv')
        
        df = pd.DataFrame()
        shap_values_df = shap_values_df.transpose()
        xch4_apriori_values = shap_values_df['xch4_apriori']
        cwave_values = ((xch4_apriori_values - xch4_apriori_values.min()) / (xch4_apriori_values.max() - xch4_apriori_values.min()) * 255).astype(int)
        df['cwave'] = cwave_values
        df['wavewave'] = df.index
        df.to_csv(cwave_wavewave_prefix + f'cwaves_decision_month_{i+1}.csv', index=False)


def fig05_06(monthly_cropdata_path, output_path):
    """
    Analyzes crop data to compare predicted values across different crop types and urban areas, and performs t-tests to evaluate statistical differences.

    This function performs the following steps:
    1. **Data Preparation**:
       - Defines categories of crops based on water intensity and drought resistance, as well as non-agricultural and urban areas.
       - Loads crop data for each month from the provided path.

    2. **Data Filtering and Analysis**:
       - Filters the data into subsets based on crop categories and calculates the mean of predicted values for each category.
       - Performs t-tests to compare the predicted values between different crop categories and urban areas.

    3. **Results Compilation**:
       - Stores the average predicted values and t-test p-values in a DataFrame.
       - Saves the DataFrame to a CSV file at the specified output path.

    Args:
        monthly_cropdata_path (str): Path to the directory containing monthly crop data CSV files.
        output_path (str): Path to save the resulting CSV file with the analysis results.

    Returns:
        None: The function saves the results to a CSV file and does not return any value.

    Notes:
        - Water intensive crops, drought resistant crops, non-agricultural areas, and urban areas are pre-defined lists.
        - T-tests are performed using a sample of 500 points for each comparison to avoid large-sample bias.
        - The function assumes that `settings.sample_size` is defined for sampling during t-tests.
    """
    # These crops were selected based on what is largely farmed around the Denver-Julesburg basin
    # to perform this analysis elsewhere, identify different crops used and their water intensity.
    water_intensive_crops = [1, 36, 41, 37]
    drought_resistant_crops = [24, 29, 42, 4, 5, 205, 28, 6, 53]
    non_agricultural = [58, 59, 60, 62, 63, 64, 65, 70, 81, 82, 83, 87, 88, 92, 111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152,
                     176, 190, 195]
    urban = [122, 123, 124]
    comb = water_intensive_crops + drought_resistant_crops + non_agricultural + urban
    other_agricultural = [x for x in range(1, 255) if x not in comb]
    
    resultdf = pd.DataFrame(columns=['month', 'wi_score', 'dr_score', 'other_ag', 'non_ag', 'urban',
                                     't_wi_dr', 't_wi_non', 't_dr_non', 't_wi_other', 't_dr_other', 't_dr_urban'])
    for month in range(1, 13):
        df = pd.read_csv(monthly_cropdata_path + 'colocated_cropdata_'+str(month).zfill(2)+'.csv')
        
        wi_subset = df[df['value'].isin(water_intensive_crops)]
        dr_subset = df[df['value'].isin(drought_resistant_crops)]
        other_subset = df[df['value'].isin(other_agricultural)]
        non_subset = df[df['value'].isin(non_agricultural)]
        urban_subset = df[df['value'].isin(urban)]
        # Urban area is small, so we will use all the points for the t-test
        urb_num = len(urban_subset)
        
        wi_corr_ave = wi_subset['predicted_values'].mean()
        dr_corr_ave = dr_subset['predicted_values'].mean()
        other_corr_ave = other_subset['predicted_values'].mean()
        non_corr_ave = non_subset['predicted_values'].mean()
        urban_corr_ave = urban_subset['predicted_values'].mean()
        
        # Only 'sample_size' number of points used because the thousands of points total would give a large-sample bias to the t-tests
        # by hugely increasing the N value. 500 chosen arbitrarily.
        s = settings.sample_size
        t_wi_dr = ttest_ind(wi_subset['predicted_values'].sample(s), dr_subset['predicted_values'].sample(s), nan_policy='omit')
        t_wi_non = ttest_ind(wi_subset['predicted_values'].sample(s), non_subset['predicted_values'].sample(s), nan_policy='omit')
        t_dr_non = ttest_ind(dr_subset['predicted_values'].sample(s), non_subset['predicted_values'].sample(s), nan_policy='omit')
        t_wi_other = ttest_ind(wi_subset['predicted_values'].sample(s), other_subset['predicted_values'].sample(s), nan_policy='omit')
        t_dr_other = ttest_ind(dr_subset['predicted_values'].sample(s), other_subset['predicted_values'].sample(s), nan_policy='omit')
        t_dr_urban = ttest_ind(dr_subset['predicted_values'].sample(urb_num), urban_subset['predicted_values'].sample(urb_num), nan_policy='omit')

        resultdf.loc[len(resultdf)] = [month, wi_corr_ave, dr_corr_ave, other_corr_ave, non_corr_ave, urban_corr_ave,
                                       t_wi_dr.pvalue, t_wi_non.pvalue, t_dr_non.pvalue, t_wi_other.pvalue, t_dr_other.pvalue, t_dr_urban.pvalue]
        
    resultdf.to_csv(output_path)


def fig07(winter_output_prefix, summer_output_prefix, z_var='xch4_corrected'):
    """
    Processes and averages crop data across two 3-month periods (winter and summer) and saves the results to CSV files.

    This function performs the following steps:
    1. **Load and Process Monthly Data**:
       - Loads data from CSV files for each month.
       - Assigns each point to a grid cell based on latitude and longitude.
       - Computes the average value for each grid cell and variable.

    2. **Average Data Across Months**:
       - Combines data from the specified months.
       - Computes the average values for each grid cell over the 3-month period.

    3. **Save Results**:
       - Saves the averaged data for the winter and summer periods to separate CSV files.

    Args:
        winter_output_prefix (str): Prefix for the output CSV file for the winter period.
        summer_output_prefix (str): Prefix for the output CSV file for the summer period.
        z_var (str): The variable name to process and average. Defaults to 'xch4_corrected'.

    Returns:
        None: The function saves the processed data to CSV files and does not return any value.

    Notes:
        - The function assumes that `settings.resolution` and `settings.colocated_cropdata_output_path` are defined.
        - The latitude and longitude grid binning is based on `settings.resolution`.
    """
    def load_and_process_monthly_data(file_paths, z_var):
        monthly_dfs = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            print(df.columns)
            df = df[['x', 'y', z_var]]
            
            # Assign each point to a grid cell
            df['lat_bin'] = np.floor(df['x'] / settings.resolution) * settings.resolution
            df['lon_bin'] = np.floor(df['y'] / settings.resolution) * settings.resolution
            
            # Group by the grid cells and compute the average for each cell
            grid_df = df.groupby(['lat_bin', 'lon_bin']).mean().reset_index()
            
            # Drop the latitude and longitude columns used for grouping
            grid_df = grid_df.drop(columns=['x', 'y'])
            
            monthly_dfs.append(grid_df)
        
        return monthly_dfs

    def average_monthly_data(monthly_dfs):
        combined_df = pd.concat(monthly_dfs)
        
        # Group by the grid cells and compute the average for the 3-month period
        avg_df = combined_df.groupby(['lat_bin', 'lon_bin']).mean().reset_index()
        
        return avg_df
    
    # Define the file paths for the months you want to average together
    months_1_2_3 = [settings.colocated_cropdata_output_path + 'colocated_cropdata_01.csv',
                    settings.colocated_cropdata_output_path + 'colocated_cropdata_02.csv',
                    settings.colocated_cropdata_output_path + 'colocated_cropdata_03.csv']
    
    months_7_8_9 = [settings.colocated_cropdata_output_path + 'colocated_cropdata_07.csv',
                    settings.colocated_cropdata_output_path + 'colocated_cropdata_08.csv',
                    settings.colocated_cropdata_output_path + 'colocated_cropdata_09.csv']
    
    # Process and average data for each 3-month period
    monthly_dfs_1_2_3 = load_and_process_monthly_data(months_1_2_3, z_var)
    avg_df_1_2_3 = average_monthly_data(monthly_dfs_1_2_3)
    
    monthly_dfs_7_8_9 = load_and_process_monthly_data(months_7_8_9, z_var)
    avg_df_7_8_9 = average_monthly_data(monthly_dfs_7_8_9)

    avg_df_1_2_3.to_csv(winter_output_prefix +'_'+ z_var + '.csv', index=False)
    avg_df_7_8_9.to_csv(summer_output_prefix +'_'+ z_var + '.csv', index=False)