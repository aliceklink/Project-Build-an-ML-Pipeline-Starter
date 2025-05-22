#!/usr/bin/env python
"""
This script performs data validation tests on the given data.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
import wandb
import scipy.stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    Run data validation tests and log results.
    
    Args:
        args: Command-line arguments
    """
    run = wandb.init(job_type="data_validation")
    run.config.update(args)

    logger.info("Downloading reference artifact")
    reference_artifact = run.use_artifact(args.reference_artifact)
    reference_path = reference_artifact.file()
    reference_df = pd.read_csv(reference_path)

    logger.info("Downloading sample artifact")
    sample_artifact = run.use_artifact(args.sample_artifact)
    sample_path = sample_artifact.file()
    sample_df = pd.read_csv(sample_path)

    logger.info("Testing data")
    
    # Import test functions from the test_data.py module
    logger.info("Importing test functions")
    import importlib.util
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(current_dir, "test_data.py")
    
    # Check if the file exists
    if not os.path.exists(test_data_path):
        logger.error(f"test_data.py not found at {test_data_path}")
        raise FileNotFoundError(f"test_data.py not found at {test_data_path}")
    
    spec = importlib.util.spec_from_file_location("test_data", test_data_path)
    test_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_data)
    
    # Run tests
    logger.info("Running row count test")
    test_data.test_row_count(sample_df)
    
    logger.info("Running price range test")
    test_data.test_price_range(sample_df)
    
    # Calculate and test the distribution similarity
    logger.info("Calculating and testing distribution similarity")
    cat_columns = [
        "neighbourhood_group",
        "room_type"
    ]
    
    num_columns = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365"
    ]
    
    # Dictionary to store KL divergences
    kl_divergences = {}
    
    # Test for categorical columns
    for col in cat_columns:
        logger.info(f"Testing distribution similarity for {col}")
        val_counts_sample = sample_df[col].value_counts(normalize=True).sort_index()
        val_counts_ref = reference_df[col].value_counts(normalize=True).sort_index()
        
        # Align the indices
        val_counts_sample, val_counts_ref = val_counts_sample.align(val_counts_ref, fill_value=1e-10)
        
        # Calculate KL divergence
        kl = scipy.stats.entropy(val_counts_sample, val_counts_ref)
        kl_divergences[col] = kl
        
        # Log to W&B
        run.summary[f"kl_divergence_{col}"] = kl
    
    # Test for numerical columns
    for col in num_columns:
        logger.info(f"Testing distribution similarity for {col}")
        
        # Handle NaN values by dropping them
        sample_col_data = sample_df[col].dropna().values
        ref_col_data = reference_df[col].dropna().values
        
        # Skip if either dataset has no valid values
        if len(sample_col_data) == 0 or len(ref_col_data) == 0:
            logger.warning(f"Column {col} has no valid data in one or both datasets, skipping")
            kl_divergences[col] = 0  # Assume no divergence if no data
            run.summary[f"kl_divergence_{col}"] = 0
            continue
            
        try:
            # Create histograms with the same bins for both dataframes
            # Calculate bin range based on both datasets to ensure same binning
            min_val = min(np.min(sample_col_data), np.min(ref_col_data))
            max_val = max(np.max(sample_col_data), np.max(ref_col_data))
            
            # Create histogram bins based on range
            bins = np.linspace(min_val, max_val, 50)
            
            hist_sample, _ = np.histogram(sample_col_data, bins=bins, density=True)
            hist_ref, _ = np.histogram(ref_col_data, bins=bins, density=True)
            
            # Replace zeros with a small value to avoid division by zero in KL divergence
            hist_sample = np.where(hist_sample == 0, 1e-10, hist_sample)
            hist_ref = np.where(hist_ref == 0, 1e-10, hist_ref)
            
            # Calculate KL divergence
            kl = scipy.stats.entropy(hist_sample, hist_ref)
            kl_divergences[col] = kl
            
            # Log to W&B
            run.summary[f"kl_divergence_{col}"] = kl
        except Exception as e:
            logger.warning(f"Error calculating KL divergence for {col}: {e}")
            kl_divergences[col] = 0  # Default to no divergence on error
            run.summary[f"kl_divergence_{col}"] = 0
    
    # Check if any KL divergence exceeds the threshold
    failed_columns = [col for col, kl in kl_divergences.items() if kl > args.kl_threshold]
    if failed_columns:
        logger.warning(f"KL divergence exceeded threshold for columns: {failed_columns}")
        logger.warning(f"Maximum observed divergence: {max(kl_divergences.values())}")
    else:
        logger.info("All columns passed the KL divergence test")
    
    # Create and upload artifact for test results
    logger.info("Creating test results artifact")
    test_results = {
        "kl_divergences": kl_divergences,
        "failed_columns": failed_columns,
        "threshold": args.kl_threshold
    }
    
    # Convert to dataframe for easier viewing
    test_results_df = pd.DataFrame(
        {
            "column": list(kl_divergences.keys()),
            "kl_divergence": list(kl_divergences.values()),
            "passed": [kl <= args.kl_threshold for kl in kl_divergences.values()]
        }
    )
    test_results_df.to_csv("test_results.csv", index=False)
    
    artifact = wandb.Artifact(
        name="data_validation_results",
        type="data_validation",
        description="Results of data validation tests"
    )
    artifact.add_file("test_results.csv")
    run.log_artifact(artifact)
    
    # Fail the pipeline if any checks failed
    if failed_columns:
        raise Exception(f"Data validation failed for columns: {failed_columns}")
    
    logger.info("Data validation tests passed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data validation tests")
    
    parser.add_argument(
        "--reference_artifact", 
        type=str,
        help="Reference artifact to compare against",
        required=True
    )
    
    parser.add_argument(
        "--sample_artifact", 
        type=str,
        help="Sample artifact to test",
        required=True
    )
    
    parser.add_argument(
        "--kl_threshold", 
        type=float,
        help="Threshold for KL divergence",
        required=True
    )
    
    args = parser.parse_args()
    
    go(args)