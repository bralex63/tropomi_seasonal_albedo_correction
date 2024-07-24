# -*- coding: utf-8 -*-
# The python code in this file was written by Alexander C. Bradley

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import optuna
import random
import warnings
from flaml import AutoML
from keras.callbacks import CSVLogger

import settings

warnings.filterwarnings("ignore", message="Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.")
tf.config.run_functions_eagerly(True)
    
    
def train_monthly_models(storage, annual_model, prefix):
    """
    Train and save machine learning models for each month based on Optuna-optimized hyperparameters.

    This function iterates through each month of the year, loads the best hyperparameters from an Optuna study for that month,
    creates and trains a machine learning model using these hyperparameters, and then saves the trained model to a specified location.

    Parameters:
    storage (str): The storage URL for the Optuna database. This URL should point to where the Optuna studies are stored.
    annual_model (keras.Model): The base model architecture or pre-trained model used as the starting point for training.
    prefix (str): The prefix used for naming the saved model files. The final model file names will be constructed by appending
                  the month number to this prefix.

    Returns:
    None

    Notes:
    - The function assumes that the `load_and_preprocess_data` function is defined elsewhere and correctly handles data loading
      and preprocessing.
    - The `create_model` function should also be defined elsewhere and is responsible for creating a model based on the provided
      hyperparameters and base model.
    - The trained model is saved with a filename that includes the month number (e.g., "model_01", "model_02", ..., "model_12").
    """
    tf.random.set_seed(settings.random_state)
    for month in range(1, 13):
        # Connect to the SQLite database
        study_name = settings.study_name_prefix + str(month)
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        # Get the best trial
        best_trial = study.best_trial

        # Extract best hyperparameters
        best_params = best_trial.params
        print(best_params)
        learning_rate = best_params['learning_rate']
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        frozen_layers = best_params['frozen_layers']

        # Load and preprocess data for this month
        X_train, y_train, X_test, y_test, features = load_and_preprocess_data(settings.training_fraction, month)
        
        # Create and compile the model
        model = create_model(None, learning_rate, frozen_layers, annual_model)

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        
        # Save the trained model
        model.save(prefix+'_'+str(month).zfill(2))


def optuna_hyperparameter_training_monthly(storage, annual_model):
    """
    Perform Optuna hyperparameter optimization for training models on a monthly basis.

    This function creates an Optuna study for each month of the year to find the best hyperparameters for training a model.
    It optimizes the hyperparameters by minimizing the objective function and then prints the best hyperparameters found.

    Parameters:
    storage (str): The storage URL for the Optuna database. This URL should point to where the Optuna studies are stored.
    annual_model (keras.Model): The base model architecture or pre-trained model used as the starting point for training.
    
    Returns:
    None
    
    Notes:
    - The function assumes that the `objective` function is defined elsewhere and is used to evaluate the performance of the model
      with the given hyperparameters.
    - The `settings` object should contain the following attributes:
      - `random_state`: The random seed for reproducibility.
      - `study_name_prefix`: The prefix to be used for naming the Optuna studies.
      - `training_fraction`: The fraction of the dataset used for training.
      - `num_training_iterations`: The number of trials for the Optuna optimization.
    - The best parameters for each month are printed to the console.
    """
    tf.random.set_seed(settings.random_state)
    global month
    for month in range(1, 13):
        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='minimize', storage=storage, study_name=settings.study_name_prefix+str(month))
        objective_monthly_models = lambda trial: objective(trial, annual_model, settings.training_fraction)
        study.optimize(objective_monthly_models, n_trials=settings.num_training_iterations)
        #study.optimize(objective, n_trials=num_trials)
    
        # Print the best parameters found
        best_params = study.best_params
        print("Best parameters:", best_params)
    

def create_model(trial, learning_rate, frozen_layers, annual_model):
    """
    Create a Keras model based on a pre-trained annual model with additional custom layers.

    This function loads a base model from a specified file, freezes a portion of its layers, and adds additional custom
    layers on top of it. The model is then compiled with the specified learning rate.

    Parameters:
    trial (optuna.Trial): The Optuna trial object (currently unused, but included for compatibility with hyperparameter
                          optimization workflows).
    learning_rate (float): The learning rate for the Adam optimizer.
    frozen_layers (int): The number of layers from the base model to freeze during training.
    annual_model (str): The file path or name of the pre-trained annual model to load.

    Returns:
    tf.keras.Model: The compiled Keras model with additional layers on top of the base model.

    Notes:
    - The base model is loaded using `tf.keras.models.load_model`, which assumes the model is saved in the standard Keras
      format.
    - The function creates a new `tf.keras.Sequential` model with the base model as the first layer, followed by three
      dense layers with ReLU activation, and a final dense layer with a single output.
    - The model is compiled with the Adam optimizer and a mean squared error loss function, and metrics include mean squared
      error (MSE).
    - The `trial` parameter is included to maintain compatibility with Optuna optimization, but is not used in this function.
    """
    base_model = tf.keras.models.load_model(annual_model)

    # Freeze layers
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False
    
    # Add new layers on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    # Compile the model with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    return model


def objective(trial, annual_model, training_fraction):
    """
    Objective function for hyperparameter optimization with Optuna.

    This function defines the hyperparameters to be optimized, creates and trains a Keras model using these
    hyperparameters, and evaluates the model on a validation set. The validation loss is returned as the objective
    value to be minimized.

    Parameters:
    trial (optuna.Trial): The Optuna trial object, used to suggest hyperparameters for the model.
    annual_model (str): The file path or name of the pre-trained annual model to be used as the base model.
    training_fraction (float): The fraction of the data to be used for training. This determines how much of the
                               dataset will be used for training versus validation.

    Returns:
    float: The validation loss of the model, which Optuna will use to assess and optimize the hyperparameters.

    Notes:
    - The function suggests hyperparameters using Optuna's `trial` object:
        - `learning_rate`: A float between 1e-7 and 0.3 (log scale).
        - `epochs`: An integer between 10 and 100.
        - `batch_size`: One of the values from [16, 32, 64, 128].
        - `frozen_layers`: An integer between 0 and 5 indicating the number of layers from the base model to freeze.
    - The function creates a model with these hyperparameters, trains it on the training data, and evaluates its performance
      on the validation data.
    - The validation loss (the first value of the evaluation score) is returned, which Optuna uses to determine the quality
      of the hyperparameters and guide the optimization process.
    """
    # Define the parameters to be optimized
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 0.3, log=True)
    epochs = trial.suggest_int("epochs", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    frozen_layers = trial.suggest_int('frozen_layers', 0, 5)

    # Create the model
    model = create_model(trial, learning_rate, frozen_layers, annual_model)
    X_train, y_train, X_test, y_test, features = load_and_preprocess_data(training_fraction, month)
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model on the validation set
    score = model.evaluate(X_test, y_test)

    # Return the validation loss
    return score[0]
    
    
#################################################################
#################################################################
#################################################################
#################################################################    
    

def train_annual_model(storage, annual_model_name):
    """
    Trains an annual model using hyperparameters optimized by Optuna and saves the trained model.

    This function performs the following steps:
    1. Loads the best hyperparameters for the annual model from an Optuna study.
    2. Prepares the data for training and validation.
    3. Creates and compiles the model using the best hyperparameters.
    4. Trains the model on the training data.
    5. Saves the trained model to the specified file.

    Parameters:
    storage (str): The storage URL for the Optuna database where the study is stored.
    annual_model_name (str): The file name or path where the trained annual model will be saved.

    Returns:
    None

    Notes:
    - The function uses a study named `settings.study_name_prefix + '_annual'` to load the Optuna study.
    - It extracts the best hyperparameters for `learning_rate`, `epochs`, `batch_size`, and `loss_function`.
    - The function assumes that `load_and_preprocess_data` is a function that returns the training and testing data.
    - The `build_annual_model` function is used to create the model with the specified optimizer and loss function.
    - The model is trained with the specified number of epochs and batch size, and its performance is validated using the test set.
    """
    tf.random.set_seed(settings.random_state)
    study = optuna.load_study(study_name=settings.study_name_prefix + '_annual', storage=storage)
    # Get the best trial
    best_trial = study.best_trial

    # Extract best hyperparameters
    best_params = best_trial.params
    print(best_params)
    learning_rate = best_params['learning_rate']
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    loss_function = best_params['loss_function']

    # Load and preprocess data for this month
    X_train, y_train, X_test, y_test, features = load_and_preprocess_data(settings.training_fraction)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Create and compile the model
    model = build_annual_model(X_train, optimizer, loss_function)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save(annual_model_name)
    
    
def optuna_hyperparameter_tuning_annual(storage):
    """
    Tunes hyperparameters for an annual model using Optuna optimization.

    This function performs the following steps:
    1. Creates an Optuna study for hyperparameter tuning with the goal of minimizing the objective function.
    2. Defines the objective function to be optimized.
    3. Runs the optimization process for a specified number of trials.
    4. Prints the best hyperparameters found during the optimization process.

    Parameters:
    storage (str): The storage URL for the Optuna database where the study will be stored and accessed.

    Returns:
    None

    Notes:
    - The function uses a study named `settings.study_name_prefix+'_annual'` for tuning the hyperparameters.
    - It optimizes the hyperparameters by minimizing the objective function defined by `annual_objective`.
    - `settings.training_fraction` and `settings.num_training_iterations` are used for the optimization process.
    - The function assumes that `annual_objective` is a predefined function that takes a trial and a training fraction as arguments and returns a value to be minimized.
    """
    tf.random.set_seed(settings.random_state)
    study = optuna.create_study(direction='minimize', storage=storage, study_name=settings.study_name_prefix+'_annual')
    objective_annual_model = lambda trial: annual_objective(trial, settings.training_fraction)
    study.optimize(objective_annual_model, n_trials=settings.num_training_iterations)
    study.optimize(annual_objective, n_trials=settings.num_training_iterations)
    
    best_params = study.best_params
    
    print("Best Parameters:")
    print(best_params)

    
def build_annual_model(X_train, optimizer, loss_function):
    """
    Builds and compiles a neural network model for training.

    This function creates a neural network model with multiple hidden layers and compiles it using
    the provided optimizer and loss function. The model is designed for regression tasks.

    Parameters:
    X_train (np.ndarray or pd.DataFrame): The input features of the training data. This should include all variables
        except for the ground truth (target values). The shape of `X_train` determines the input shape of the model.
    optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training the model. Examples include Adam,
        SGD, etc.
    loss_function (str or callable): The loss function to use for training the model. Examples include 'mean_squared_error',
        'mean_absolute_error', or a custom loss function.

    Returns:
    tf.keras.Model: A compiled TensorFlow Keras model ready for training.
    """
    # Create a neural network model with 2 hidden layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['MSE'])
    
    return model
    
    
def annual_objective(trial, training_fraction):
    """
    Objective function for hyperparameter optimization using Optuna.

    This function defines a set of hyperparameters for training a neural network model, trains the model using these
    hyperparameters, and evaluates the model to return a metric that Optuna will use to guide the optimization process.

    Parameters:
    trial (optuna.Trial): An Optuna trial object that provides methods to sample hyperparameters.
    training_fraction (float): Fraction of the data to use for training. It is used to load and preprocess the data.

    Returns:
    float: The mean squared error (MSE) of the model on the test set. This is the metric that Optuna aims to minimize.

    Notes:
    - `learning_rate` is sampled from a log scale between 1e-7 and 0.1.
    - `batch_size` is sampled from an integer range between 5 and 100.
    - `epochs` is sampled from a categorical set [16, 32, 64, 128, 256].
    - `loss_function` is sampled from categorical options ['MSE', 'MAE'].
    """
    # Define hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 0.0000001, 0.1, log=True)
    batch_size = trial.suggest_int('batch_size', 5, 100)
    epochs = trial.suggest_categorical('epochs', [16, 32, 64, 128, 256])
    loss_function = trial.suggest_categorical('loss_function', ['MSE', 'MAE'])
    
    X_train, y_train, X_test, y_test, features = load_and_preprocess_data(training_fraction)
    
    # Add other hyperparameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Build the model with the suggested hyperparameters
    model = build_annual_model(X_train, optimizer, loss_function)
    
    # Compile and train the model, return the validation metric to optimize
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    eval_results = model.evaluate(X_test, y_test)
    MSE = eval_results[1]
    
    return MSE


#################################################################
#################################################################
#################################################################
#################################################################  

def load_and_preprocess_data(training_data_fraction, month=None):
    """
    Loads and preprocesses the training and testing data for model training.

    This function reads the TROPOMI - GOSAT point pairs data, standardizes the features, and splits the data into 
    training and testing sets. It also handles monthly filtering if specified.

    Parameters:
    training_data_fraction (float): Fraction of the total data to be used for training.
    month (int, optional): If specified, filters the data to include only records from the given month.

    Returns:
    tuple: A tuple containing:
        - X_train (pd.DataFrame): Features for the training set.
        - y_train (pd.Series): Ground truth values for the training set.
        - X_test (pd.DataFrame): Features for the test set.
        - y_test (pd.Series): Ground truth values for the test set.
        - features (list): List of feature names used in the training set.
    """
    # Load in TROPOMI - GOSAT point pairs
    training_data = pd.read_csv(settings.training_data_path, encoding='latin-1')
    training_data.drop(columns=['geometry'], inplace=True)
    training_data_size = int(training_data_fraction * len(training_data))
    # Standardize the variables
    training_data, z_stats = z_score_standardization(training_data)
    # Save the standardization scheme for later standards and getting real numbers back out
    with open(settings.z_stats_path, 'w+b') as f:
        pickle.dump(z_stats, f)
    # Un-standardize the variables "month" and "ground_truth" for accurate modeling
    # training_data['ground_truth'] = (training_data['ground_truth']*z_stats['ground_truth'][1]) + z_stats['ground_truth'][0]
    training_data['month'] = (training_data['month']*z_stats['month'][1]) + z_stats['month'][0]
    training_data_only = training_data.sample(training_data_size, random_state=42)
    testing_data_only = training_data.merge(training_data_only.drop_duplicates(), how='left', indicator=True)
    testing_data_only = testing_data_only[testing_data_only['_merge']=='left_only'].drop(columns=['_merge'])

    if month is not None:
        training_data_only = training_data_only[training_data_only['month']==month]
        testing_data_only = testing_data_only[testing_data_only['month']==month]

    cols = settings.training_data_columns
    cols.append('delta_tropomi_gosat')
    training_data_only = training_data_only[cols]
    testing_data_only = testing_data_only[cols]

    X_train, y_train, features = split_data(training_data_only, 0)
    X_test, y_test, test_features = split_data(testing_data_only, 0)
    
    return X_train, y_train, X_test, y_test, features

    
def split_data(df, test_size):
    """
    Splits the dataframe into features and target variables for model training.

    This function drops rows with missing values, separates the dataframe into feature variables and target variables,
    and returns them as numpy arrays.

    Parameters:
    df (pd.DataFrame): DataFrame containing the dataset with features and target variable.
    test_size (float): Proportion of the data to be used as test set. This parameter is currently not used in the function.

    Returns:
    tuple: A tuple containing:
        - X (np.ndarray): Array of feature variables.
        - y (np.ndarray): Array of target variable.
        - df_features (Index): List of feature names.
    """
    val_df = df
    val_df.dropna(inplace=True)

    features = val_df.drop(columns=['delta_tropomi_gosat'], axis=1)
    target = val_df['delta_tropomi_gosat']
    df_features = features.columns
    
    # Convert features and target to numpy arrays
    X = features.values
    y = target.values
    
    return X, y, df_features
    
    
def z_score_standardization(df):
    """
    Standardizes the features in a DataFrame using z-score normalization, leaving the target variable unchanged.

    For each feature, this function computes the z-score by subtracting the mean and dividing by the standard deviation.
    It returns the standardized DataFrame and a dictionary containing the mean and standard deviation for each feature.

    Parameters:
    df (pd.DataFrame): DataFrame with features to be standardized and target variable.

    Returns:
    tuple: A tuple containing:
        - z_score_df (pd.DataFrame): DataFrame with standardized features.
        - z_score_data_dict (dict): Dictionary where keys are feature names and values are lists of [mean, std] for each feature.
    """
    z_score_df = pd.DataFrame()
    z_score_data_dict = {}
    for column in df.columns:
        if column == 'delta_tropomi_gosat':
            z_score_df[column] = df[column]
            continue
        avg = df[column].mean()
        std = df[column].std()
        z_score_df[column] = (df[column] - avg)/std
        z_score_data_dict[column] = [avg, std]
    return z_score_df, z_score_data_dict


def known_z_score(df, z_stats):
    """
    Standardizes the features in a DataFrame using precomputed z-score statistics.

    This function uses provided mean and standard deviation values to apply z-score normalization to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with features to be standardized.
    z_stats (dict): Dictionary where keys are feature names and values are lists of [mean, std] used for standardization.

    Returns:
    pd.DataFrame: DataFrame with standardized features.
    """
    z_score_df = pd.DataFrame()
    for column in df.columns:
        avg = z_stats[column][0]
        std = z_stats[column][1]
        z_score_df[column] = (df[column] - avg)/std
    return z_score_df


