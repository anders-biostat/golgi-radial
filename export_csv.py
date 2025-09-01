#!/usr/bin/env python3
"""
CSV Export Script
Generate comprehensive CSV file with one row per cell containing data from both analysis modes.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse

# Import functions from the main analysis script
import sys
sys.path.append('.')
from golgi_analysis import run_analysis, find_70pct_quantile, parse_filename

def run_both_modes():
    """Run analysis in both modes and return combined results"""
    print("Running analysis in both modes...")
    
    # Temporarily modify the analysis script to run both modes
    import golgi_analysis as ga
    
    # Save original mode
    original_mode = ga.ANALYSIS_MODE
    original_plots = {
        'ECDF': ga.GENERATE_ECDF_PLOTS,
        'BEESWARM': ga.GENERATE_BEESWARM_PLOTS,
        'VIOLIN': ga.GENERATE_VIOLIN_PLOTS
    }
    
    # Disable plots for faster processing
    ga.GENERATE_ECDF_PLOTS = False
    ga.GENERATE_BEESWARM_PLOTS = False  
    ga.GENERATE_VIOLIN_PLOTS = False
    
    results = {}
    
    # Run golgi mode
    print("\n--- Running Golgi-centered analysis ---")
    ga.ANALYSIS_MODE = 'golgi'
    results['golgi'] = ga.run_analysis()
    
    # Run centrosome mode
    print("\n--- Running Centrosome-centered analysis ---")
    ga.ANALYSIS_MODE = 'centrosome'
    results['centrosome'] = ga.run_analysis()
    
    # Restore original settings
    ga.ANALYSIS_MODE = original_mode
    ga.GENERATE_ECDF_PLOTS = original_plots['ECDF']
    ga.GENERATE_BEESWARM_PLOTS = original_plots['BEESWARM']
    ga.GENERATE_VIOLIN_PLOTS = original_plots['VIOLIN']
    
    return results

def create_mode_csv(results, mode_name, output_file=None):
    """Create CSV file for a single analysis mode"""
    
    if results.empty:
        print(f"ERROR: No results for {mode_name} mode")
        return False
    
    if output_file is None:
        output_file = f"cell_analysis_{mode_name}.csv"
    
    print(f"\nCreating {mode_name} mode CSV:")
    print(f"  Cells: {len(results)}")
    
    # Create the dataframe with desired columns
    csv_data = []
    
    for idx, row in results.iterrows():
        # Calculate standard deviation
        std_dev = np.sqrt(row['variance']) if not np.isnan(row['variance']) else np.nan
        
        csv_row = {
            'file_basename': row['basename'],
            'cell_index': row['cell_idx'],
            'frame': row['frame'],
            'condition': row['cond'],
            'genotype': row['genotype'],
            '70pct_quantile': row['70pct_quantile'],
            'std_deviation': std_dev,
            'filename': row['filename']  # Keep full filename for reference
        }
        
        csv_data.append(csv_row)
    
    # Create DataFrame
    csv_df = pd.DataFrame(csv_data)
    
    # Sort by filename and cell index for better readability
    csv_df = csv_df.sort_values(['filename', 'cell_index']).reset_index(drop=True)
    
    # Save to CSV
    csv_df.to_csv(output_file, index=False)
    
    print(f"  Output file: {output_file}")
    print(f"  Columns: {list(csv_df.columns)}")
    
    return True

def create_combined_csv(results_golgi, results_centrosome, output_file="cell_analysis_combined.csv"):
    """Create single CSV with data from both analysis modes"""
    
    if results_golgi.empty or results_centrosome.empty:
        print("ERROR: One or both analysis modes returned no results")
        return False
    
    print(f"\nCombining results from both modes:")
    print(f"  Golgi mode: {len(results_golgi)} cells")
    print(f"  Centrosome mode: {len(results_centrosome)} cells")
    
    # Create a unique identifier for each cell (filename + cell_idx)
    results_golgi['cell_id'] = results_golgi['filename'] + '_' + results_golgi['cell_idx'].astype(str)
    results_centrosome['cell_id'] = results_centrosome['filename'] + '_' + results_centrosome['cell_idx'].astype(str)
    
    # Check if cell IDs match between modes
    golgi_ids = set(results_golgi['cell_id'])
    centrosome_ids = set(results_centrosome['cell_id'])
    common_ids = golgi_ids.intersection(centrosome_ids)
    
    print(f"  Common cells: {len(common_ids)}")
    
    if len(common_ids) != len(golgi_ids) or len(common_ids) != len(centrosome_ids):
        print("  WARNING: Cell counts don't match between modes!")
    
    # Use common cells and merge the data
    results_golgi_common = results_golgi[results_golgi['cell_id'].isin(common_ids)].copy()
    results_centrosome_common = results_centrosome[results_centrosome['cell_id'].isin(common_ids)].copy()
    
    # Sort both by cell_id to ensure proper alignment
    results_golgi_common = results_golgi_common.sort_values('cell_id').reset_index(drop=True)
    results_centrosome_common = results_centrosome_common.sort_values('cell_id').reset_index(drop=True)
    
    # Create the combined dataframe
    combined_data = []
    
    for idx, row_golgi in results_golgi_common.iterrows():
        row_centrosome = results_centrosome_common.iloc[idx]
        
        # Calculate standard deviations
        golgi_std = np.sqrt(row_golgi['variance']) if not np.isnan(row_golgi['variance']) else np.nan
        centrosome_std = np.sqrt(row_centrosome['variance']) if not np.isnan(row_centrosome['variance']) else np.nan
        
        # Create combined row with clean column names
        combined_row = {
            'file_basename': row_golgi['basename'],
            'cell_index': row_golgi['cell_idx'],
            'frame': row_golgi['frame'],
            'condition': row_golgi['cond'],
            'genotype': row_golgi['genotype'],
            'golgi_70pct_quantile': row_golgi['70pct_quantile'],
            'golgi_std_deviation': golgi_std,
            'centrosome_70pct_quantile': row_centrosome['70pct_quantile'],
            'centrosome_std_deviation': centrosome_std,
            'filename': row_golgi['filename']  # Keep full filename for reference
        }
        
        combined_data.append(combined_row)
    
    # Create DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Sort by filename and cell index for better readability
    combined_df = combined_df.sort_values(['filename', 'cell_index']).reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCSV export completed:")
    print(f"  Output file: {output_file}")
    print(f"  Total cells: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  Conditions: {combined_df['condition'].value_counts().to_dict()}")
    print(f"  Genotypes: {combined_df['genotype'].value_counts().to_dict()}")
    
    # Show first few rows
    print(f"\nFirst 3 rows:")
    print(combined_df.head(3)[['file_basename', 'cell_index', 'condition', 'genotype', 
                               'golgi_70pct_quantile', 'golgi_std_deviation',
                               'centrosome_70pct_quantile', 'centrosome_std_deviation']].to_string(index=False))
    
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Export cell analysis data to CSV")
    parser.add_argument("--output", "-o", default="cell_analysis_combined.csv", 
                       help="Output CSV filename (default: cell_analysis_combined.csv)")
    parser.add_argument("--use-existing", action="store_true", 
                       help="Use existing pickle files instead of rerunning analysis")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CELL ANALYSIS CSV EXPORT")
    print("=" * 70)
    
    if args.use_existing:
        # Try to load existing results
        try:
            print("Loading existing results from pickle files...")
            results_golgi = pd.read_pickle("results_golgi.pkl")
            results_centrosome = pd.read_pickle("results_centrosome.pkl")
            print("  ✓ Successfully loaded existing results")
        except FileNotFoundError as e:
            print(f"  ✗ Could not load existing results: {e}")
            print("  Running fresh analysis instead...")
            results = run_both_modes()
            results_golgi = results['golgi']
            results_centrosome = results['centrosome']
    else:
        # Run fresh analysis
        results = run_both_modes()
        results_golgi = results['golgi']
        results_centrosome = results['centrosome']
    
    # Create combined CSV file
    success = create_combined_csv(results_golgi, results_centrosome, args.output)
    
    if success:
        print("\n" + "=" * 70)
        print("CSV EXPORT SUCCESSFUL")
        print(f"Generated: {args.output}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("CSV EXPORT FAILED")
        print("=" * 70)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())