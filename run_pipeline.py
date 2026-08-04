import argparse
from pathlib import Path
from src.golgi_analysis.segmentation import run_cellpose_segmentation, filter_masks
from src.golgi_analysis.analysis import run_analysis_pipeline
from src.golgi_analysis.postprocessing import remove_outliers_iqr
from src.golgi_analysis.plotting import (plot_mean_curves, plot_beeswarm_cross_genotype, plot_beeswarm_by_genotype)

def main():
    parser = argparse.ArgumentParser(description="Golgi & Centrosome Radial Distribution Analysis Pipeline")
    
    parser.add_argument("--data-dir", type=str, required=True, help="Base experiment folder containing rawImages/")
    parser.add_argument("--model-path", type=str, required=True, help="Path to custom Cellpose model")
    parser.add_argument("--mode", type=str, choices=['golgi', 'centrosome'], default='golgi', help="Analysis target mode")
    parser.add_argument("--min-cell-size", type=int, default=500, help="Minimum pixel area threshold for cells")
    parser.add_argument("--run-segmentation", action="store_true", help="Run Cellpose segmentation and filtering")
    
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    raw_images_dir = base_dir / "rawImages"
    raw_masks_dir = base_dir / "masks"
    filtered_masks_dir = base_dir / "filtered_masks"
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 & 2: Cellpose Segmentation and Mask Filtering
    if args.run_segmentation:
        print("\n--- STAGE 1: Running Cellpose Segmentation ---")
        run_cellpose_segmentation(raw_images_dir, raw_masks_dir, args.model_path)
        
        print("\n--- STAGE 2: Filtering Cell Masks ---")
        filter_masks(raw_masks_dir, filtered_masks_dir, min_size=args.min_cell_size)

    # Stage 3: Image Feature Measurement
    print("\n--- STAGE 3: Extracting Radial Metrics ---")
    results = run_analysis_pipeline(raw_images_dir, filtered_masks_dir, analysis_mode=args.mode)

    if not results.empty:
        # Save raw results
        raw_csv = output_dir / f"results_raw_{args.mode}.csv"
        results.to_csv(raw_csv)

        # Stage 4: Outlier Filtering
        print("\n--- STAGE 4: Removing Outliers (IQR) ---")
        filtered_results = remove_outliers_iqr(results, col_measure='70pct_quantile')
        filtered_csv = output_dir / f"results_filtered_{args.mode}.csv"
        filtered_results.to_csv(filtered_csv)

        # Stage 5: Plotting
        print("\n--- STAGE 5: Generating Plots ---")
        # 1. Mean ECDF curves per condition comparing genotypes
        plot_mean_curves(filtered_results, output_dir, mode=args.mode)

        # 2. Beeswarm comparing genotypes per condition
        plot_beeswarm_cross_genotype(filtered_results, output_dir, mode=args.mode)

        # 3. Beeswarm/Violin plots per genotype across conditions
        plot_beeswarm_by_genotype(filtered_results, output_dir, mode=args.mode, plot_violins=True)

        print(f"\n✅ Done! Analysis complete. Results saved to: {output_dir}")
    else:
        print("❌ No valid results were extracted.")

if __name__ == "__main__":
    main()