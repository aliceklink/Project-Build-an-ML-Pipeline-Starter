#!/usr/bin/env python
"""
This script trains a Random Forest model on the NYC Airbnb dataset
"""
import argparse
import logging
import os
import shutil

import mlflow
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Train a Random Forest model and save it as an MLflow model
    
    Args:
        args: Command line arguments
    """
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and cast parameters
    logger.info(f"Loading RF config from {args.rf_config}")
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    
    # Fix random_state if present
    if "random_state" in rf_config:
        rf_config["random_state"] = int(rf_config["random_state"])

    # Convert other parameters to int as needed
    for param in ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf"]:
        if param in rf_config:
            rf_config[param] = int(rf_config[param])

    logger.info(f"RF config: {rf_config}")
    
    # Get the training dataset
    logger.info(f"Loading training data from {args.trainval_artifact}")
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    logger.info(f"Training data saved to {trainval_local_path}")
    
    df = pd.read_csv(trainval_local_path)
    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Training data columns: {df.columns.tolist()}")
    
    # Drop rows with any missing values to simplify the pipeline
    df = df.dropna()
    logger.info(f"Shape after dropping missing values: {df.shape}")
    
    # Extract target from the dataset
    X = df.copy()
    y = X.pop("price")
    
    # Check for any remaining columns with missing values
    missing_cols = X.columns[X.isna().any()].tolist()
    if missing_cols:
        logger.warning(f"Columns with missing values: {missing_cols}")
    
    # Debugging
    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")
    
    # Split training and validation
    logger.info("Splitting data into train and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_size,
        stratify=X[args.stratify_by] if args.stratify_by != "none" else None,
        random_state=args.random_seed
    )
    logger.info(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Define categorical and numerical columns
    categorical_cols = ["neighbourhood_group", "room_type"]
    numerical_cols = [
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365"
    ]
    
    # Verify columns exist in the DataFrame
    for col in categorical_cols + numerical_cols:
        if col not in X.columns:
            logger.warning(f"Column {col} not found in DataFrame")
    
    logger.info("Building the pipeline")
    
    # Create a simple preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('text', TfidfVectorizer(max_features=args.max_tfidf_features), 'name')
        ],
        remainder='drop'
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(**rf_config))
    ])
    
    # Train the model
    logger.info("Training the model")
    try:
        # Convert DataFrames to numpy arrays to avoid pandas-specific issues
        # For the text column, extract it separately
        X_train_cat_num = X_train[categorical_cols + numerical_cols].values
        X_train_text = X_train['name'].values
        
        # Create new DataFrame with proper structure
        X_train_combined = pd.DataFrame({
            'name': X_train_text
        })
        
        # Add categorical and numerical columns
        for i, col in enumerate(categorical_cols + numerical_cols):
            X_train_combined[col] = X_train_cat_num[:, i]
        
        # Fit the pipeline
        pipeline.fit(X_train_combined, y_train)
        
        logger.info("Model trained successfully")
        
        # Prepare validation data in the same way
        X_val_cat_num = X_val[categorical_cols + numerical_cols].values
        X_val_text = X_val['name'].values
        
        X_val_combined = pd.DataFrame({
            'name': X_val_text
        })
        
        for i, col in enumerate(categorical_cols + numerical_cols):
            X_val_combined[col] = X_val_cat_num[:, i]
        
        # Score the model
        logger.info("Evaluating the model")
        r_squared = pipeline.score(X_val_combined, y_val)
        logger.info(f"R-squared on validation set: {r_squared}")
        
        # Make predictions
        y_pred = pipeline.predict(X_val_combined)
        mae = mean_absolute_error(y_val, y_pred)
        logger.info(f"MAE on validation set: {mae}")
        
        # Log metrics
        run.summary["r2"] = r_squared
        run.summary["mae"] = mae
        
        # Export the model
        logger.info("Exporting the model")
        export_path = "random_forest_dir"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
            
        mlflow.sklearn.save_model(
            pipeline,
            export_path
        )
        
        # Create artifact for the model export
        logger.info(f"Creating artifact: {args.output_artifact}")
        artifact = wandb.Artifact(
            name=args.output_artifact,
            type="model_export",
            description="Random Forest model export"
        )
        artifact.add_dir(export_path)
        run.log_artifact(artifact)
        logger.info("Artifact created and logged")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    # Removed the run.wait() line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model")
    
    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset",
        required=True
    )
    
    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split (fraction)",
        required=True
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
        required=True
    )
    
    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        required=True
    )
    
    parser.add_argument(
        "--rf_config",
        type=str,
        help="Random forest configuration. JSON file expected",
        required=True
    )
    
    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        help="Maximum number of features for TFIDF",
        required=True
    )
    
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True
    )

    args = parser.parse_args()

    go(args)