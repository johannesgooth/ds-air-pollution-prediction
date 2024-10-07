import sys
# setting path
sys.path.append('../')
import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.metrics import r2_score
from src.evaluation.evaluation_functions import *
from src.visualization.visualization_functions import *
import pandas as pd

import numpy as np
import tensorflow as tf 
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from scikeras.wrappers import KerasRegressor

# Create the custom RMSE scoring object
rmse_score = make_scorer(rmse_scorer, greater_is_better=False)

def train_and_evaluate_model(model, param_grid, preprocessing_pipeline, search_type='grid', n_iter=10, X_train=None, X_test=None, y_train=None, y_test=None, scoring='neg_root_mean_squared_error'):
    """
    Train and evaluate the model using either GridSearchCV or RandomizedSearchCV.

    Parameters:
    model: The regression model to be used.
    param_grid: Dictionary of hyperparameters for tuning.
    preprocessing_pipeline: The preprocessing steps to be applied to the data.
    search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
    n_iter: Number of iterations for RandomizedSearchCV, ignored if search_type is 'grid'.
    X_train, X_test, y_train, y_test: Training and testing data.
    scoring: Metric for evaluation, e.g., 'neg_root_mean_squared_error'.
    
    Returns:
    Trained model and test results.
    """
    
    # Create the pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', model)
    ])
    
    # Choose between GridSearchCV and RandomizedSearchCV
    if search_type == 'grid':
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring)
    elif search_type == 'random':
        search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring=scoring, n_iter=n_iter, random_state=42)

    # Train the model
    search.fit(X_train, y_train)
    
    # Predict on training set
    y_train_pred = search.predict(X_train)
    
    # Calculate training metrics
    rmse_train = rmse_scorer(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    n_train = X_train.shape[0]
    p_train = X_train.shape[1]
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p_train - 1)
    
    # Start a single MLflow run for both training and test metrics
    with mlflow.start_run(run_name="Training_and_Testing_Evaluation"):
        # Log model type and search type
        mlflow.log_param('model_type', model.__class__.__name__)  # Log model type
        mlflow.log_param('search_type', search_type)  # Log search type
        
        # Log the best hyperparameters found by GridSearchCV or RandomizedSearchCV
        mlflow.log_params(search.best_params_)
        
        # Log training metrics
        mlflow.log_metric("RMSE_train", rmse_train)
        mlflow.log_metric("R2_train", r2_train)
        mlflow.log_metric("Adjusted_R2_train", adj_r2_train)
        
        # Generate training error analysis plot
        error_analysis_plot(y_train, y_train_pred)
        
        print(f"Best Parameters: {search.best_params_}")
        print(f"Training RMSE: {rmse_train}")
        print(f"Training R²: {r2_train}")
        print(f"Training Adjusted R²: {adj_r2_train}")
    
        # Predict on the test set using the best model
        best_model = search.best_estimator_
        y_test_pred = best_model.predict(X_test)

        # Calculate test metrics
        rmse_test = rmse_scorer(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        n_test = X_test.shape[0]
        p_test = X_test.shape[1]
        adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)
        
        # Log test metrics
        mlflow.log_metric("RMSE_test", rmse_test)
        mlflow.log_metric("R2_test", r2_test)
        mlflow.log_metric("Adjusted_R2_test", adj_r2_test)
        
        # Generate test error analysis plot
        error_analysis_plot(y_test, y_test_pred)
        
        print(f"Test RMSE: {rmse_test}")
        print(f"Test R²: {r2_test}")
        print(f"Test Adjusted R²: {adj_r2_test}")

    return best_model, {'rmse_test': rmse_test, 'r2_test': r2_test, 'adj_r2_test': adj_r2_test}, y_test_pred

def create_mlp_model(n_neurons=64, learning_rate=0.001, input_shape=None, **kwargs):
    print(f"Received parameters: n_neurons={n_neurons}, learning_rate={learning_rate}, input_shape={input_shape}")
    model = tf.keras.Sequential()
    if input_shape is not None:
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu', input_shape=input_shape))
    else:
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model

def get_mlflow_metrics_from_custom_tracking(experiment_name):
    """
    Retrieves metrics, model type, and search type for all models logged in a given MLflow experiment 
    with a custom tracking URI.
    
    Parameters:
    experiment_name: The name of the MLflow experiment.

    Returns:
    DataFrame with metrics, model type, and search type collected from MLflow.
    """
    # Set the custom tracking URI
    tracking_uri = os.path.join(os.getcwd(), "..", "models", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
    
    experiment_id = experiment.experiment_id
    
    # Search all runs for the given experiment
    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

    # Define the metrics we want to collect
    metrics_to_collect = ['RMSE_train', 'R2_train', 'Adjusted_R2_train', 'RMSE_test', 'R2_test', 'Adjusted_R2_test']

    # Create a list to store results
    metrics_list = []

    # Loop through each run
    for _, run in runs_df.iterrows():
        run_id = run['run_id']
        run_data = mlflow.get_run(run_id).data

        # Collect metrics for each run
        run_metrics = {metric: run_data.metrics.get(metric, None) for metric in metrics_to_collect}
        
        # Collect model type and search type
        run_metrics['Model Type'] = run_data.params.get('model_type', 'N/A')  # Model type can be logged during run
        run_metrics['Search Type'] = run_data.params.get('search_type', 'N/A')  # Search type can be logged during run
        
        # Collect Run ID and Best Params for reference
        run_metrics['Run ID'] = run_id
        run_metrics['Best Params'] = run_data.params
        
        metrics_list.append(run_metrics)

    # Convert list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    return metrics_df