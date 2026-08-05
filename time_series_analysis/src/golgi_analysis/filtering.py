"""Mask size filtering and dataset-level IQR outlier removal."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import tifffile
from .plotting import plot_outlier_check


def run_mask_filtering(
    mask_dir: str,
    filt_dir: str,
    min_size_threshold: int = 500,
    max_frames: int = 30
):
    print("\n" + "=" * 60)
    print("STEP 3 — Mask Filtering")
    print("=" * 60)

    input_dir, output_dir = Path(mask_dir), Path(filt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted([f for f in input_dir.glob("*.tif") if "_filtered" not in f.name])

    for file_path in tif_files:
        print(f"\n--- {file_path.name} ---")
        mask_stack = tifffile.imread(str(file_path))[:max_frames]
        n_frames = mask_stack.shape[0]

        all_ids = np.unique(mask_stack)
        all_ids = all_ids[all_ids > 0]
        bad_ids = set()
        kept_cell_sizes = {}
        valid_t_count = sum(1 for t in range(n_frames) if np.max(mask_stack[t]) > 0)

        for cid in all_ids:
            sizes = [np.sum(mask_stack[t] == cid) for t in range(n_frames)]
            active_sizes = [s for s in sizes if s > 0]

            if not active_sizes or np.median(active_sizes) < min_size_threshold or len(active_sizes) < valid_t_count:
                bad_ids.add(cid)
                continue

            on_border = False
            for t in range(n_frames):
                f = mask_stack[t]
                if cid in f[0, :] or cid in f[-1, :] or cid in f[:, 0] or cid in f[:, -1]:
                    on_border = True
                    break
            if on_border:
                bad_ids.add(cid)
                continue

            kept_cell_sizes[cid] = np.median(active_sizes)

        good_ids = [cid for cid in all_ids if cid not in bad_ids]
        print(f"  > Total: {len(all_ids)} | Rejected: {len(bad_ids)} | Kept: {len(good_ids)}")

        max_id = int(np.max(mask_stack))
        lookup_table = np.zeros(max_id + 1, dtype=mask_stack.dtype)
        for new_id, old_id in enumerate(good_ids, start=1):
            lookup_table[old_id] = new_id

        filtered_stack = lookup_table[mask_stack]
        output_path = output_dir / f"{file_path.stem}_filtered.tif"
        tifffile.imwrite(str(output_path), filtered_stack.astype(np.uint16), imagej=True, metadata={"axes": "TYX"})

    print("\nStep 3 complete.")


def filter_cells_by_iqr(df: pd.DataFrame, column: str, multiplier: float) -> pd.DataFrame:
    """Drops entire cell tracks if any frame exceeds the IQR threshold."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    bad_cells = df.loc[outlier_mask, "global_cell_idx"].unique()
    good_mask = ~df["global_cell_idx"].isin(bad_cells)

    dropped_frames = (~good_mask).sum()
    print(f"[{column}] Bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    print(f"  -> Found {len(bad_cells)} cells with outliers. Dropping their {dropped_frames} total frames.")

    return df[good_mask]


def clean_results_csv(input_csv: str, output_dir: str, iqr_multiplier: float = 1.5):
    """Loads raw results CSV, performs full-track IQR filtering, exports diagnostics, and saves clean output."""
    print(f"\nLoading data from {input_csv} for IQR cleaning...")
    df = pd.read_csv(input_csv)

    if "variance" in df.columns and "std_dev" not in df.columns:
        df["std_dev"] = np.sqrt(df["variance"])
    if "timeseries" in df.columns and "cell_idx" in df.columns:
        df["global_cell_idx"] = df["timeseries"].astype(str) + "_cell_" + df["cell_idx"].astype(str)
    elif "filename" in df.columns and "cell_idx" in df.columns:
        df["global_cell_idx"] = df["filename"].astype(str) + "_cell_" + df["cell_idx"].astype(str)

    df_clean = filter_cells_by_iqr(df, "cell_area", iqr_multiplier)
    df_clean = filter_cells_by_iqr(df_clean, "std_dev", iqr_multiplier)

    input_filename_root = os.path.splitext(os.path.basename(input_csv))[0]
    output_csv_path = os.path.join(output_dir, f"{input_filename_root}_cleaned.csv")
    plot_path = os.path.join(output_dir, f"{input_filename_root}_outlier_check.png")

    plot_outlier_check(df, df_clean, plot_path)
    df_clean.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset saved → {output_csv_path}")
    return output_csv_path