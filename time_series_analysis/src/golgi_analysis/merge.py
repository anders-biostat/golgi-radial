"""Batch directory CSV merging utilities."""

import glob
import os
import pandas as pd


def merge_csv_files(folder_path: str, output_filename: str = "merged_results.csv") -> str:
    """Concatenates all CSV files in a given folder into a single unified CSV file."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f) != output_filename]

    print(f"\nMerging {len(csv_files)} CSV file(s) from {folder_path}...")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files available to merge in {folder_path}")

    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    output_path = os.path.join(folder_path, output_filename)
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully merged data → {output_path}")
    return output_path