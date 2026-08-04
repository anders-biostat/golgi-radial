from pathlib import Path
import pandas as pd

def remove_outliers_iqr(
    df: pd.DataFrame, 
    col_group: str = 'cond', 
    col_measure: str = '70pct_quantile'
) -> pd.DataFrame:
    """Filter out values outside of 1.5 * IQR per condition group."""
    grouped = df.groupby(col_group)[col_measure]
    
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_df = df[(df[col_measure] >= lower_bound) & (df[col_measure] <= upper_bound)].copy()
    print(f"Outliers removed: {len(df) - len(filtered_df)} / {len(df)} rows")
    return filtered_df


def merge_csv_files(input_dir: Path, output_file: Path):
    """Combine all stage-result CSV files in a directory into one master file."""
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    frames = [pd.read_csv(f, index_col=0) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_file, index=True)
    print(f"Successfully combined {len(csv_files)} files into {output_file}")