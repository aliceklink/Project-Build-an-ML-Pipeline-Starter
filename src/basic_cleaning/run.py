#!/usr/bin/env python
"""
This script performs basic data cleaning on the input dataset.
"""
import argparse
import logging
import os
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Clean the dataset and upload the resulting artifact
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Filter by price
    logger.info("Filtering data by price")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Filter out locations outside of NYC proper boundaries
    logger.info("Filtering out locations outside of NYC proper boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    # Write the clean dataframe to a CSV file
    logger.info("Writing clean data to CSV")
    df.to_csv("clean_sample.csv", index=False)
    
    # Create and upload the artifact
    logger.info("Creating and uploading artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    
    # Log basic statistics
    logger.info("Logging statistics")
    run.log({
        "n_total": len(pd.read_csv(artifact_local_path)),
        "n_price_filtered": len(df),
        "min_price": df["price"].min(),
        "max_price": df["price"].max(),
        "median_price": df["price"].median()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A basic data cleaning step")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to include",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to include",
        required=True
    )

    args = parser.parse_args()

    go(args)