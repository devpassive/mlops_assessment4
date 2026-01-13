#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import pandas as pd
import wandb
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Started cleaning data")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    logger.info(f"Total raw rows loaded {len(df)}")
    # cleaning trailing spaces if any
    df.columns = df.columns.str.strip()

    # ensure price is numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # filter price outliers
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")


    # geographic bounds
    geo_idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[geo_idx].copy()
    logger.info(f"Remaining rows after filtering and cleaning: {len(df)}")

    # 3) Save
    output_file = "clean_sample.csv"
    df.to_csv(output_file, index=False)

    # 4) Log output artifact
    out_art = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    out_art.add_file(output_file)

    run.log_artifact(out_art)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()

    go(args)
