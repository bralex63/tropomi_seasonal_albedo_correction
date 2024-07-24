"""
Title: Deep Transfer Learning Method for Seasonal TROPOMI XCH4 Albedo Correction

Description:
This Python code accompanies the paper "Deep Transfer Learning Method for Seasonal TROPOMI XCH4 Albedo Correction."
It implements the methods and analyses discussed in the publication and produces the data required to construct the
figures therein.

Authors:
Alexander C. Bradley, Cooperative Institute for Research in Environmental Sciences, University of Colorado, Boulder, 
    CO, 80309; Chemistry Department, University of Colorado, Boulder, CO, 80309
Barbara Dix, Cooperative Institute for Research in Environmental Sciences, University of Colorado, Boulder, 
    CO, 80309
Fergus Mackenzie, BlueSky Resources, Boulder CO, 80302, USA
J. Pepijn Veefkind, Royal Netherlands Meteorological Institute (KNMI), De Bilt, The Netherlands
    Delft University of Technology, Delft, The Netherlands
Joost A. de Gouw, Cooperative Institute for Research in Environmental Sciences, University of Colorado, Boulder, 
    CO, 80309; Chemistry Department, University of Colorado, Boulder, CO, 80309

Citation:
If you use this code in your research, please cite the following paper:
    Paper submitted to Atmospheric Measurement Techniques, this will be updated with a DOI once assigned

Contact:
For any questions or issues regarding this code, please contact Alex Bradley at albr9412@colorado.edu.
"""

# The python code in this file was written by Alexander C. Bradley

from model_data_prep import regrid_gosat_files, regrid_tropomi_files, combine_gosat_tropomi, prepare_data_paths
from train_models_4 import optuna_hyperparameter_tuning_annual, optuna_hyperparameter_training_monthly, train_annual_model, train_monthly_models
from correct_albedo_and_regrid import correct_albedo_and_regrid
from process_ag_data import process_ag_data, correlate_corrected_data_with_agriculture
from ml_paper_figures import fig02, fig03, fig04, fig05_06, fig07

import settings


if __name__ == "__main__":

    # Prepare file structure: check the settings.py and change any paths before running this function
    prepare_data_paths()

##############################################################################
# Preparing the data for training
##############################################################################
        # regridding University of Leicester Full-Physics retrieval (UoL_FP) GOSAT data
        # regridding TROPOMI data to the same grid
        # Combining the regridded TROPOMI and GOSAT data on the same time and spatial scales
        #   and calculating delta_tropomi_gosat

    UoL_FP_GOSAT_loc = settings.UoL_FP_GOSAT_loc
    regridded_gosat_loc = settings.regridded_gosat_loc

    TROPOMI_data_loc = settings.TROPOMI_data_loc
    regridded_tropomi_loc = settings.regridded_tropomi_loc

    combined_tropomi_gosat_output_loc = settings.training_data_path

    resolution = settings.resolution               # in degrees
    bounds = settings.bounds    # [longmin, latmin, longmax, latmax] Rectangular only
 
    # These can be run one at a time, or you may uncomment all three of them at once
    #regrid_gosat_files(UoL_FP_GOSAT_loc, resolution, regridded_gosat_loc, bounds)
    #regrid_tropomi_files(TROPOMI_data_loc, resolution, regridded_tropomi_loc, bounds)
    #combine_gosat_tropomi(regridded_tropomi_loc, regridded_gosat_loc, combined_tropomi_gosat_output_loc)

    # Be sure to comment (#) the above three functions before moving on!

    # Figures
    # At this point, Figure 2 can be constructed
    fig02_output_path = settings.fig02_output_path
    #fig02(combined_tropomi_gosat_output_loc, fig02_output_path)

##############################################################################
# Training the models
##############################################################################
    # First optimize the annual model hyperparameters
    # Then train the annual model
    # Next optimize the monthly model hyperparameters (different hyperparameters for each month)
    # Finally, train the monthly models
    
    training_data_path = settings.training_data_path
    storage = settings.storage

    annual_model_name = settings.annual_model_name
    monthly_models_prefix = settings.monthly_models_prefix

    # the following functions should only be uncommented and run one at a time!
    #optuna_hyperparameter_tuning_annual(storage)
    #train_annual_model(storage, annual_model_name)
    #optuna_hyperparameter_training_monthly(storage, annual_model_name)
    #train_monthly_models(storage, annual_model_name, monthly_models_prefix)

    # Be sure to comment (#) the above four functions before moving on!   

    # Figures
    # In order to construct figure 3, Balasus et al. data must be processed first:
    BLND_data_path = settings.BLND_data_path
    regridded_BLND_path = settings.regridded_BLND_path
    combined_BLND_gosat_path = settings.combined_BLND_gosat_path
    #regrid_tropomi_files(BLND_data_path, resolution, regridded_BLND_path, bounds)
    #combine_gosat_tropomi(regridded_BLND_path, regridded_gosat_loc, combined_BLND_gosat_path, True)

    #  At this point, figures 3 and 4 can be constructed
    fig03_data_output_path = settings.fig03_data_output_path
    #fig03(fig03_data_output_path)

    decision_plot_data_prefix = settings.decision_plot_data_prefix
    cwave_wavewave_prefix = settings.cwave_wavewave_prefix
    shap_values_pickle_path = settings.shap_values_pickle_path
    expected_values_pickle_path = settings.expected_values_pickle_path
    fig04_overwrite = settings.fig04_overwrite
    #fig04(decision_plot_data_prefix, cwave_wavewave_prefix, shap_values_pickle_path, expected_values_pickle_path, fig04_overwrite)


##############################################################################
# Co-location with land-use data
##############################################################################
    # must be downloaded in the locations and time periods of your choosing. https://nassgeodata.gmu.edu/CropScape/
    # specify the projection as "Degrees Lat/Lon, WGS84 Datum", extract the .tif file and specify its path
    regridded_albedo_corrected_data_path = settings.regridded_albedo_corrected_data_path

    ag_raster_path = settings.ag_raster_path
    downsampled_raster_output_path = settings.downsampled_raster_output_path
    output_pickle_path = settings.output_pickle_path
    colocated_cropdata_output_path = settings.colocated_cropdata_output_path

    #correct_albedo_and_regrid(TROPOMI_data_loc, regridded_albedo_corrected_data_path)
    #process_ag_data(ag_raster_path, downsampled_raster_output_path, output_pickle_path)
    #correlate_corrected_data_with_agriculture(regridded_albedo_corrected_data_path, output_pickle_path, colocated_cropdata_output_path)

    #  At this point, figures 5, 6, and 7 can be constructed:
    fig05_06_output_path = settings.fig05_06_output_path
    #fig05_06(colocated_cropdata_output_path, fig05_06_output_path)

    winter_output_prefix = settings.winter_output_prefix
    summer_output_prefix = settings.summer_output_prefix
    # for variable in ['xch4_corrected', 'xch4_builtin', 'surface_albedo_SWIR']:
    #     fig07(winter_output_prefix, summer_output_prefix, variable)