# The python code in this file was written by Alexander C. Bradley
# The settings are organized the same way as the albedo_correction_main file

# If you want the default file organization, this is the only output path you'll
# need to change. You will still need to change input paths for the code to find your data
target_directory = 'm:/testing_albedo_correction_code/'

# Input data paths:
UoL_FP_GOSAT_loc = 'm:/UoL_FP_GOSAT/' # Downloaded from CEDA archive https://doi.org/10.5285/18ef8247f52a4cb6a14013f8235cc1eb
TROPOMI_data_loc = 'm:/CH4_RPRO/' # Downloaded from Copernicus datapace https://browser.dataspace.copernicus.eu
BLND_data_path = 'm:/BLND_Balasus/'     # Balasus et al. data downloaded from https://dataverse.harvard.edu/dataverse/blended-tropomi-gosat-methane
ag_raster_path = 'm:/testing_albedo_correction_code/CDL_2019_clip_20240723125025_515932112.tif'
# Agricultural raster data are aquired from https://nassgeodata.gmu.edu/CropScape/ 


##############################################################################
# Preparing the data for training
##############################################################################
# The following path definitions are for data that are downloaded from the specified location

resolution = 0.1                # Spatial resolution GOSAT and TROPOMI are to be regridded to (probably 0.1)
bounds = [-105, 34, -95, 42]    # Where in the world is the data you are interested in? [longmin, latmin, longmax, latmax] 
training_data_columns = ['solar_zenith_angle',  # These are the column names of the TROPOMI data used to train the models
                        'relative_azimuth_angle',   # (You probably don't want to change this)
                        'ground_pixel',
                        'xch4', 
                        'xch4_corrected',
                        'surface_pressure',
                        'pressure_interval',
                        'surface_altitude',
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
                        'chi_square_NIR']

# The following path definitions are for data that are calculated and reprocessed in the code
# and saved to your computer for later uses

# regridded GOSAT and TROPOMI datasets are saved into a specified FOLDER
regridded_gosat_loc = target_directory + 'regridded_UoL_GOSAT/'
regridded_tropomi_loc = target_directory + 'regridded_TROPOMI_RPRO/'

# z_stats is a pickled file (.pkl) that contains standardization information
z_stats_path = target_directory + 'z_stats.pkl'

# Training data is saved to a specified .csv FILE
training_data_path = target_directory + 'combined_gosat_tropomi_0p1.csv'

fig02_output_path = target_directory + 'fig02.jpg'


##############################################################################
# Training the models
##############################################################################
# model paths have no executable at the end (.csv etc) because they are saved as folders
model_folder_path = target_directory + 'models/'
monthly_model_folder_path = target_directory + 'models/monthly_models/'
annual_model_name = model_folder_path + 'base_model_22072024'
monthly_models_prefix = monthly_model_folder_path + 'ensemble_model_22072024' # appropriate month number will be appended to end ex: '_03'

storage = "sqlite:///m:/testing_albedo_correction_code/hyperparameter_training_data.sqlite3" # SQLite file contains training hyperparameters
study_name = "best_annual_hyperparameters"  # This is just a name within the SQLite file
study_name_prefix = "optuna_parameters"

# Optuna training parameters
num_training_iterations = 2 # How many training sessions to determine the best hyperparameters
training_fraction = 0.8     # What fraction of data to use for training (probably between 0.5 and 0.9)
random_state = 42           # For reproducibility, specify an integer random state
tccon_offset = 9.2          # Average difference of worldwide measurement for GOSAT and TROPOMI
                            # (you probably don't want to change this)

# Fig03 settings

regridded_BLND_path = target_directory + 'regridded_BLND/'
combined_BLND_gosat_path = target_directory +'combined_gosat_BLND_0p1.csv'
fig03_data_output_path = target_directory + 'fig03.csv'

# Fig04 settings
fig04_folder_path = target_directory + 'fig04_data/'
decision_plot_data_prefix = fig04_folder_path
cwave_wavewave_prefix = fig04_folder_path
shap_values_pickle_path = target_directory + 'shap_values.pkl'
expected_values_pickle_path = target_directory + 'expected_values.pkl'
fig04_overwrite = False

##############################################################################
# Co-location with land-use data
##############################################################################

regridded_albedo_corrected_data_path = target_directory + 'regridded_albedo_corrected_tropomi/'
downsampled_raster_output_path = target_directory + 'downsampled.tif'
output_pickle_path = target_directory + 'downsampled_crops.pkl'
colocated_cropdata_output_path = target_directory + 'colocated_cropdata/'
downsampling_factor = 100

# Figure 5 and 6:
sample_size = 500
fig05_06_output_path = target_directory + 'fig05_06.csv'

# Figure 7
winter_output_prefix = target_directory + 'fig07_winter'
summer_output_prefix = target_directory + 'fig07_summer'

