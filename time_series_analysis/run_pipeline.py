#!/usr/bin/env python3
"""
Golgi & Centrosome Radial Analysis Pipeline (Time-Series Edition)
================================================================
CLI entry point to execute individual steps or the complete time-series workflow.

Usage Examples:
    # Run full pipeline end-to-end:
    python run_pipeline.py --data-dir "./EX3/250KO" --mode centrosome --step all

    # Run only core math analysis:
    python run_pipeline.py --data-dir "./EX3/250KO" --mode golgi --step analyze

    # Run IQR clean-up on existing results:
    python run_pipeline.py --data-dir "./EX3/250KO" --mode centrosome --step clean

    # Generate line plot for a specific metric:
    python run_pipeline.py --data-dir "./EX3/250KO" --mode centrosome --metric 70pct_quantile --step plot
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

from src.golgi_analysis import (
    run_illumination_correction,
    run_segmentation,
    run_mask_filtering,
    process_tiff_file,
    parse_filename,
    find_70pct_quantile,
    clean_results_csv,
    merge_csv_files,
    plot_timecourse,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time-Series Golgi and Centrosome Radial Dispersion Analysis Pipeline"
    )
    
    # Path settings
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the base experimental directory containing 'raw/' folder"
    )
    
    # Mode & metric settings
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centrosome", "golgi"],
        default="centrosome",
        help="Analysis mode: 'centrosome' (centrosome-centered) or 'golgi' (organelle-centered)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["std_dev", "70pct_quantile"],
        default="std_dev",
        help="Dispersion metric to plot: 'std_dev' (spatial std dev) or '70pct_quantile' (70%% radius)"
    )
    
    # Workflow step execution control
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "illum", "segment", "filter", "analyze", "clean", "merge", "plot"],
        default="all",
        help="Pipeline step to execute"
    )
    
    # Channel configuration
    parser.add_argument("--nucleus-ch", type=int, default=0, help="Channel index for Nucleus (default: 0)")
    parser.add_argument("--golgi-ch", type=int, default=1, help="Channel index for Golgi (default: 1)")
    parser.add_argument("--centrosome-ch", type=int, default=2, help="Channel index for Centrosome (default: 2)")
    
    # Processing parameters
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum number of frames to process per movie (default: 30)")
    parser.add_argument("--min-size", type=int, default=500, help="Minimum median cell mask area in pixels (default: 500)")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR outlier filter multiplier threshold (default: 1.5)")
    parser.add_argument("--frame-interval", type=float, default=5.0, help="Frame time interval in minutes (default: 5.0)")
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir).resolve()
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        sys.exit(1)

    raw_dir = base_dir / "raw"
    illum_dir = base_dir / "illumination_correction"
    qc_dir = illum_dir / "QC_flatfields"
    mask_dir = base_dir / "masks"
    filt_dir = base_dir / "masks_filtered"
    results_dir = base_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GOLGI RADIAL TIME-SERIES PIPELINE")
    print("=" * 60)
    print(f"Base Directory : {base_dir}")
    print(f"Analysis Mode  : {args.mode.upper()}")
    print(f"Target Step    : {args.step.upper()}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: ILLUMINATION CORRECTION (BaSiC)
    # -------------------------------------------------------------------------
    if args.step in ["all", "illum"]:
        illum_files = list(illum_dir.glob("*_corrected.tif*")) if illum_dir.exists() else []
        if args.step == "all" and illum_files:
            print(f"\n[SKIPPING STEP 1] Found {len(illum_files)} flatfield-corrected file(s) in {illum_dir.name}/")
        else:
            run_illumination_correction(
                raw_dir=str(raw_dir),
                illum_dir=str(illum_dir),
                qc_dir=str(qc_dir),
                max_frames=args.max_frames
            )

    # -------------------------------------------------------------------------
    # STEP 2: SEGMENTATION & TRACKING (Cellpose + Phase Correlation)
    # -------------------------------------------------------------------------
    if args.step in ["all", "segment"]:
        mask_files = list(mask_dir.glob("*_masks.tif*")) if mask_dir.exists() else []
        if args.step == "all" and mask_files:
            print(f"\n[SKIPPING STEP 2] Found {len(mask_files)} mask stack file(s) in {mask_dir.name}/")
        else:
            run_segmentation(
                illum_dir=str(illum_dir),
                mask_dir=str(mask_dir),
                nucleus_ch=args.nucleus_ch,
                golgi_ch=args.golgi_ch,
                max_frames=args.max_frames
            )

    # -------------------------------------------------------------------------
    # STEP 3: MASK FILTERING (Size, Border & Continuity Filters)
    # -------------------------------------------------------------------------
    if args.step in ["all", "filter"]:
        filt_files = list(filt_dir.glob("*_filtered.tif*")) if filt_dir.exists() else []
        if args.step == "all" and filt_files:
            print(f"\n[SKIPPING STEP 3] Found {len(filt_files)} filtered mask file(s) in {filt_dir.name}/")
        else:
            run_mask_filtering(
                mask_dir=str(mask_dir),
                filt_dir=str(filt_dir),
                min_size_threshold=args.min_size,
                max_frames=args.max_frames
            )

    # -------------------------------------------------------------------------
    # STEP 4: CORE RADIAL MATH ANALYSIS
    # -------------------------------------------------------------------------
    raw_csv_path = results_dir / f"{args.mode}.csv"
    if args.step in ["all", "analyze"]:
        print("\n" + "=" * 60)
        print(f"STEP 4 — Core Radial Analysis (Mode: {args.mode.upper()})")
        print("=" * 60)

        tiff_files = [
            os.path.relpath(os.path.join(root, file), illum_dir)
            for root, _, files in os.walk(illum_dir) for file in files if file.endswith(".tif")
        ]

        if not tiff_files:
            print(f"Error: No TIFF files found in {illum_dir}")
            sys.exit(1)

        print(f"Found {len(tiff_files)} TIFF image file(s).")
        max_cores = max(1, os.cpu_count() - 2)
        print(f"Executing parallel calculation across {max_cores} CPU core(s)...")

        results_list = Parallel(n_jobs=max_cores)(
            delayed(process_tiff_file)(
                filename=f,
                illum_dir=str(illum_dir),
                filt_dir=str(filt_dir),
                analysis_mode=args.mode,
                max_frames=args.max_frames,
                golgi_ch=args.golgi_ch,
                centrosome_ch=args.centrosome_ch
            ) for f in tiff_files
        )

        all_results = [r for r in results_list if r is not None and not r.empty]

        if not all_results:
            print("WARNING: No analysis results generated! Please check input images and mask files.")
        else:
            combined = pd.concat(all_results, ignore_index=True)
            combined["basename"] = combined["filename"].apply(lambda x: os.path.basename(x).replace(".tif", ""))
            combined[["cond", "experiment", "timeseries"]] = combined["basename"].apply(parse_filename)

            combined["70pct_quantile"] = [
                find_70pct_quantile(row["statistic"]) if len(row["statistic"]) > 0 else float("nan")
                for _, row in combined.iterrows()
            ]

            combined.to_csv(raw_csv_path, index=False)
            print(f"\nRaw analysis results saved → {raw_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 5: OUTLIER CLEANING (Whole-Track IQR Filter)
    # -------------------------------------------------------------------------
    clean_csv_path = results_dir / f"{args.mode}_cleaned.csv"
    if args.step in ["all", "clean"]:
        if not raw_csv_path.exists():
            print(f"Error: Raw dataset '{raw_csv_path}' not found. Run '--step analyze' first.")
        else:
            clean_csv_path = clean_results_csv(
                input_csv=str(raw_csv_path),
                output_dir=str(results_dir),
                iqr_multiplier=args.iqr_multiplier
            )

    # -------------------------------------------------------------------------
    # STEP 6: DIRECTORY CSV MERGE
    # -------------------------------------------------------------------------
    if args.step in ["merge"]:
        merge_csv_files(
            folder_path=str(results_dir),
            output_filename=f"merged_{args.mode}.csv"
        )

    # -------------------------------------------------------------------------
    # STEP 7: PUBLICATION PLOTTING
    # -------------------------------------------------------------------------
    if args.step in ["all", "plot"]:
        merged_file = results_dir / f"merged_{args.mode}.csv"
        
        if merged_file.exists():
            target_csv = merged_file
        elif Path(clean_csv_path).exists():
            target_csv = clean_csv_path
        elif raw_csv_path.exists():
            target_csv = raw_csv_path
        else:
            print(f"Error: No valid result CSV file found in '{results_dir}' to generate plot.")
            sys.exit(1)

        output_plot_path = results_dir / f"{args.mode}_{args.metric}_dispersion_plot.png"
        print(f"\nGenerating publication plot using '{target_csv.name}'...")

        plot_timecourse(
            csv_path=str(target_csv),
            save_path=str(output_plot_path),
            analysis_mode=args.mode,
            metric=args.metric,
            frame_interval_min=args.frame_interval
        )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()