#!/usr/bin/env python3
"""
Unified Golgi Analysis Script
Combines all functionality from separate analysis scripts into one configurable tool.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tifffile
from scipy.ndimage import label
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

# Analysis mode: 'golgi' (Golgi-centered) or 'centrosome' (centrosome-centered)
ANALYSIS_MODE = 'centrosome'

# Plot generation flags
GENERATE_ECDF_PLOTS = True
GENERATE_BEESWARM_PLOTS = True
GENERATE_VIOLIN_PLOTS = False

# File paths
INPUT_DIR = "imgs_segm"
MASK_DIR = "mask"
OUTPUT_PREFIX = "results"

# Filename parsing (modify this when filename structure changes)
def parse_filename(basename):
    """Parse filename components - customize this function for new data"""
    # Handle the specific patterns in this dataset
    # Pattern examples:
    # F10_30min WO_CEP250KO -> frame:F10, cond:30min WO, genotype:CEP250KO
    # F3_NODRUG_CEP250KO -> frame:F3, cond:NODRUG, genotype:CEP250KO
    # F11_2hDRUG_CEP250KO -> frame:F11, cond:2hDRUG, genotype:CEP250KO
    # F9_2HWO_CEP250KO -> frame:F9, cond:2HWO, genotype:CEP250KO
    
    # Split on underscores first
    parts = re.split(r'_+', basename)
    parts = [part for part in parts if part]
    
    if len(parts) < 3:
        # Fallback to original parsing
        frame = parts[0] if len(parts) > 0 else ''
        cond = parts[1] if len(parts) > 1 else ''
        genotype = parts[2] if len(parts) > 2 else ''
        return pd.Series([frame, cond, genotype])
    
    # Extract frame (first part)
    frame = parts[0]
    
    # Extract genotype (last part, should be CEP250KO)
    genotype = parts[-1]
    
    # Everything in between is the condition
    cond_parts = parts[1:-1]
    cond = '_'.join(cond_parts)
    
    # Handle special cases where condition contains spaces
    # Convert back spaces in condition for patterns like "30min WO"
    if 'min' in cond and 'WO' in cond:
        cond = cond.replace('_WO', ' WO')
    
    return pd.Series([frame, cond, genotype])

# Plot settings
PLOT_DPI = 150
FIGURE_SIZE = (10, 8)
ECDF_XLIM = 200


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

@jit(nopython=True)
def get_distsq_to_center(img):
    """Calculate squared distances from each pixel to the weighted center of mass"""
    rows, cols = img.shape
    
    # Calculate weighted center of mass
    total_intensity = np.sum(img)
    if total_intensity == 0:
        return np.zeros_like(img)
    
    mean_row = 0.0
    mean_col = 0.0
    
    for i in range(rows):
        for j in range(cols):
            mean_row += img[i, j] * (i + 1)  # R uses 1-based indexing
            mean_col += img[i, j] * (j + 1)
    
    mean_row /= total_intensity
    mean_col /= total_intensity
    
    # Calculate squared distances
    distsq = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row)**2 + ((j + 1) - mean_col)**2
    
    return distsq


@jit(nopython=True)
def get_distsq_to_centrosome_center(golgi_img, centrosome_img):
    """Calculate squared distances from each pixel to the weighted center of mass of centrosome"""
    rows, cols = centrosome_img.shape
    
    # Calculate weighted center of mass of centrosome
    total_intensity = np.sum(centrosome_img)
    if total_intensity == 0:
        # Fallback to geometric center if no centrosome signal
        mean_row = rows / 2.0
        mean_col = cols / 2.0
    else:
        mean_row = 0.0
        mean_col = 0.0
        
        for i in range(rows):
            for j in range(cols):
                mean_row += centrosome_img[i, j] * (i + 1)
                mean_col += centrosome_img[i, j] * (j + 1)
        
        mean_row /= total_intensity
        mean_col /= total_intensity
    
    # Calculate squared distances for all pixels
    distsq = np.zeros_like(golgi_img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row)**2 + ((j + 1) - mean_col)**2
    
    return distsq


@jit(nopython=True)
def get_radial_variance(img, reference_img=None):
    """Compute radial variance (weighted average of squared distances)"""
    if reference_img is not None:
        distsq_to_cm = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq_to_cm = get_distsq_to_center(img)
    
    total_intensity = np.sum(img)
    
    if total_intensity == 0:
        return 0.0
    
    return np.sum(img * distsq_to_cm) / total_intensity


@jit(nopython=True)
def get_radial_ecdf(img, reference_img=None):
    """Create empirical cumulative distribution function of radial distances"""
    if reference_img is not None:
        distsq_to_cm = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq_to_cm = get_distsq_to_center(img)
    
    rows, cols = img.shape
    max_dim = max(rows, cols)
    
    # Calculate ECDF for each radius
    ecdf = np.zeros(max_dim)
    
    for r in range(1, max_dim + 1):
        r_squared = r * r
        cumulative_intensity = 0.0
        
        for i in range(rows):
            for j in range(cols):
                if distsq_to_cm[i, j] < r_squared:
                    cumulative_intensity += img[i, j]
        
        ecdf[r-1] = cumulative_intensity
    
    # Normalize by maximum value
    max_val = np.max(ecdf)
    if max_val > 0:
        ecdf = ecdf / max_val
    
    return ecdf


def find_70pct_quantile(ecdf):
    """Find the radius where ECDF reaches 0.7 (70%)"""
    if len(ecdf) == 0:
        return np.nan
    
    # Find first index where ECDF >= 0.7
    indices = np.where(ecdf >= 0.7)[0]
    if len(indices) == 0:
        # If never reaches 0.7, return the maximum radius
        return len(ecdf)
    else:
        # Return the radius (1-based indexing like R)
        return indices[0] + 1


# =============================================================================
# IMAGE PROCESSING PIPELINE
# =============================================================================

def process_tiff_file(filename):
    """Process a single TIFF file to extract radial statistics for each segmented cell"""
    print(f"Processing: {filename}")
    
    # Read TIFF file - read all pages separately
    full_path = os.path.join(INPUT_DIR, filename)
    print(f"  Reading file: {full_path}")
    
    # Name of Mask
    basename = os.path.basename(filename).replace(".tif", "")
    mask_name = f"{basename}_mask.tif"
    
    # Read Mask
    mask_path = os.path.join(MASK_DIR, mask_name)

    # Read Image and Mask Data
    image_data = tifffile.imread(image_path)
    mask_data = tifffile.imread(mask_path)

    # Extract channels
    img_golgi = first_img[:, :, 0].astype(np.float64)
    img_centr = first_img[:, :, 1].astype(np.float64) 
    img_dapi = first_img[:, :, 2].astype(np.float64)
    img_segm = mask_data.astype(np.float64)  
    
    print(f"  Extracted channels - Golgi: {img_golgi.shape}, Centrosome: {img_centr.shape}, Segmentation: {img_segm.shape}")
    
    # Mask are already labled 
    segm_labeled = img_segm
    num_cells = int(np.max(segm_labeled))
    
    print(f"  Found {num_cells} cells")
    
    # Calculate statistics for each cell
    statistics = []
    variances = []
    for cell_idx in range(1, num_cells + 1):
        # Create mask for current cell
        cell_mask = (segm_labeled == cell_idx).astype(np.float64)
        
        # Apply mask to images
        masked_golgi = img_golgi * cell_mask
        masked_centr = img_centr * cell_mask
        
        # Calculate radial ECDF and variance based on analysis mode
        if ANALYSIS_MODE == 'centrosome':
            cell_ecdf = get_radial_ecdf(masked_golgi, masked_centr)
            cell_variance = get_radial_variance(masked_golgi, masked_centr)
        else:  # 'golgi' mode
            cell_ecdf = get_radial_ecdf(masked_golgi)
            cell_variance = get_radial_variance(masked_golgi)
        
        statistics.append(cell_ecdf)
        variances.append(cell_variance)
    
    # Create results dataframe
    results = pd.DataFrame({
        'filename': [filename] * len(statistics),
        'cell_idx': range(1, len(statistics) + 1),
        'statistic': statistics,
        'variance': variances
    })
    
    return results


def run_analysis():
    """Main processing pipeline"""
    print(f"Running analysis in {ANALYSIS_MODE} mode")
    
    # Find all TIFF files
    tiff_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.tif'):
                rel_path = os.path.relpath(os.path.join(root, file), INPUT_DIR)
                tiff_files.append(rel_path)
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Process all files
    all_results = []
    for i, filename in enumerate(tiff_files):
        try:
            result = process_tiff_file(filename)
            if not result.empty:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Combine results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Parse filename components
        combined_results['basename'] = combined_results['filename'].apply(
            lambda x: os.path.basename(x).replace('.tif', '')
        )
        
        combined_results[['frame', 'cond', 'genotype']] = combined_results['basename'].apply(parse_filename)
        
        # Check genotype consistency
        unique_genotypes = combined_results['genotype'].unique()
        if len(unique_genotypes) > 1:
            print(f"WARNING: Multiple genotypes found: {list(unique_genotypes)}")
            print("This analysis assumes all samples have the same genotype.")
        elif len(unique_genotypes) == 1:
            print(f"Genotype: {unique_genotypes[0]}")
        
        # Calculate 70% quantiles
        quantiles_70pct = []
        for idx, row in combined_results.iterrows():
            if len(row['statistic']) > 0:
                q70 = find_70pct_quantile(row['statistic'])
                quantiles_70pct.append(q70)
            else:
                quantiles_70pct.append(np.nan)
        
        combined_results['70pct_quantile'] = quantiles_70pct
        
        return combined_results
    else:
        return pd.DataFrame()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def get_plot_title_suffix(results):
    """Get genotype info for plot titles"""
    unique_genotypes = results['genotype'].unique()
    if len(unique_genotypes) == 1 and unique_genotypes[0]:
        return f" - {unique_genotypes[0]}"
    return ""

def plot_ecdf_curves(results):
    """Generate ECDF curve plots"""
    if not GENERATE_ECDF_PLOTS:
        return
    
    print("Generating ECDF plots...")
    
    # Getting Information about cond name for file name
    genotype_info = results["genotype"].unique()[0]

    # Individual curves plot
    plt.figure(figsize=FIGURE_SIZE)
    
    conditions = results['cond'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))
    condition_colors = dict(zip(conditions, colors))
    
    # Plot all ECDF curves
    for idx, row in results.iterrows():
        if len(row['statistic']) > 0:
            ecdf = row['statistic']
            x_values = np.arange(1, len(ecdf) + 1)
            
            # Limit to x-axis range
            mask = x_values <= ECDF_XLIM
            x_plot = x_values[mask]
            y_plot = ecdf[mask] if len(ecdf) > len(x_plot) else ecdf[:len(x_plot)]
            
            plt.plot(x_plot, y_plot, 
                    color=condition_colors[row['cond']], 
                    alpha=0.3, 
                    linewidth=0.5)
    
    # Add legend
    legend_lines = []
    legend_labels = []
    for cond in conditions:
        line = plt.plot([], [], color=condition_colors[cond], linewidth=2, label=cond)[0]
        legend_lines.append(line)
        legend_labels.append(f"{cond} (n={sum(results['cond'] == cond)})")
    
    plt.legend(legend_lines, legend_labels, loc='lower right')
    plt.xlim(0, ECDF_XLIM)
    plt.ylim(0, 1)
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Cumulative Distribution')
    plt.title(f'Radial ECDF Curves by Condition ({ANALYSIS_MODE.title()} Mode){get_plot_title_suffix(results)}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'ecdf_curves_{ANALYSIS_MODE}_{genotype_info}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Mean curves with confidence bands
    plt.figure(figsize=FIGURE_SIZE)
    
    for cond in conditions:
        subset = results[results['cond'] == cond]
        
        # Collect all ECDFs for this condition
        ecdfs = []
        for _, row in subset.iterrows():
            if len(row['statistic']) > 0:
                ecdf = row['statistic']
                # Pad or truncate to ECDF_XLIM points
                if len(ecdf) >= ECDF_XLIM:
                    ecdfs.append(ecdf[:ECDF_XLIM])
                else:
                    # Pad with the last value
                    padded = np.concatenate([ecdf, np.full(ECDF_XLIM - len(ecdf), ecdf[-1] if len(ecdf) > 0 else 0)])
                    ecdfs.append(padded)
        
        if len(ecdfs) > 0:
            ecdfs_array = np.array(ecdfs)
            mean_ecdf = np.mean(ecdfs_array, axis=0)
            std_ecdf = np.std(ecdfs_array, axis=0)
            
            x_values = np.arange(1, ECDF_XLIM + 1)
            
            # Plot mean with confidence band (±2 SD)
            plt.plot(x_values, mean_ecdf, color=condition_colors[cond], linewidth=2, label=f"{cond} (n={len(ecdfs)})")
            plt.fill_between(x_values, 
                            mean_ecdf - 2 * std_ecdf, 
                            mean_ecdf + 2 * std_ecdf, 
                            color=condition_colors[cond], 
                            alpha=0.2)
    
    plt.xlim(0, ECDF_XLIM)
    plt.ylim(0, 1)
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Cumulative Distribution')
    plt.title(f'Mean Radial ECDF Curves by Condition ({ANALYSIS_MODE.title()} Mode, ±2 SD){get_plot_title_suffix(results)}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'mean_ecdf_curves_{ANALYSIS_MODE}_{genotype_info}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print("ECDF plots saved")




def plot_beeswarm_analysis(results):
    """Generate beeswarm plots for quantiles and variance"""
    if not GENERATE_BEESWARM_PLOTS:
        return
    
    print("Generating beeswarm plots...")
    
    # Remove rows with NaN values
    plot_data = results.dropna(subset=['variance', '70pct_quantile'])
    
    if plot_data.empty:
        print("No valid data for beeswarm plots")
        return
    
    # Getting Information about cond name for file name
    genotype_info = results["genotype"].unique()[0]

    # Standard deviation beeswarm
    plt.figure(figsize=FIGURE_SIZE)
    plot_data['std_dev'] = np.sqrt(plot_data['variance'])
    
    sns.swarmplot(data=plot_data, x='cond', y='std_dev', size=5, alpha=0.7)
    
    # Add mean lines
    for i, cond in enumerate(plot_data['cond'].unique()):
        subset = plot_data[plot_data['cond'] == cond]
        mean_val = subset['std_dev'].mean()
        plt.hlines(mean_val, i-0.25, i+0.25, colors='red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Condition')
    plt.ylabel('Standard Deviation (pixels)')
    plt.title(f'Radial Standard Deviation by Condition ({ANALYSIS_MODE.title()} Mode){get_plot_title_suffix(results)}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'beeswarm_std_dev_{ANALYSIS_MODE}_{genotype_info}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # 70% quantile beeswarm
    plt.figure(figsize=FIGURE_SIZE)
    
    sns.swarmplot(data=plot_data, x='cond', y='70pct_quantile', size=5, alpha=0.7)
    
    # Add mean lines
    for i, cond in enumerate(plot_data['cond'].unique()):
        subset = plot_data[plot_data['cond'] == cond]
        mean_val = subset['70pct_quantile'].mean()
        plt.hlines(mean_val, i-0.25, i+0.25, colors='red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Condition')
    plt.ylabel('70% Quantile Radius (pixels)')
    plt.title(f'70% Quantile of Radial ECDF by Condition ({ANALYSIS_MODE.title()} Mode){get_plot_title_suffix(results)}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'beeswarm_70pct_quantiles_{ANALYSIS_MODE}_{genotype_info}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print("Beeswarm plots saved")


def plot_violin_analysis(results):
    """Generate violin plots"""
    if not GENERATE_VIOLIN_PLOTS:
        return
    
    print("Generating violin plots...")
    
    # Remove rows with NaN values
    plot_data = results.dropna(subset=['70pct_quantile'])
    
    if plot_data.empty:
        print("No valid data for violin plots")
        return
    
    plt.figure(figsize=FIGURE_SIZE)
    
    sns.violinplot(data=plot_data, x='cond', y='70pct_quantile', inner='box')
    sns.swarmplot(data=plot_data, x='cond', y='70pct_quantile', size=3, alpha=0.6, color='white')
    
    plt.xlabel('Condition')
    plt.ylabel('70% Quantile Radius (pixels)')
    plt.title(f'70% Quantile of Radial ECDF by Condition ({ANALYSIS_MODE.title()} Mode){get_plot_title_suffix(results)}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'violin_70pct_quantiles_{ANALYSIS_MODE}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print("Violin plots saved")


def print_summary_statistics(results):
    """Print summary statistics"""
    print(f"\nSUMMARY STATISTICS ({ANALYSIS_MODE.upper()} MODE)")
    print("=" * 60)
    
    print(f"Processed {len(results)} cells from {results['filename'].nunique()} files")
    print(f"Conditions: {results['cond'].value_counts().to_dict()}")
    
    if 'variance' in results.columns:
        print(f"\nVariance summary by condition:")
        print(results.groupby('cond')['variance'].describe())
    
    if '70pct_quantile' in results.columns:
        valid_quantiles = results.dropna(subset=['70pct_quantile'])
        print(f"\n70% quantile summary by condition ({len(valid_quantiles)} valid measurements):")
        print(valid_quantiles.groupby('cond')['70pct_quantile'].describe())


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOLGI RADIAL DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Run analysis
    results = run_analysis()
    
    if not results.empty:
        
        # Getting Information about cond name for file name
        genotype_info = results["genotype"].unique()[0]
        
        # Save results as CSV
        output_file = f"{OUTPUT_PREFIX}_{ANALYSIS_MODE}_{genotype_info}.csv"
        results.to_csv(output_file)
        print(f"Results saved to {output_file}")
        
        # Generate all requested plots
        plot_ecdf_curves(results)
        plot_beeswarm_analysis(results)
        plot_violin_analysis(results)
        
        # Print summary
        print_summary_statistics(results)
        
    else:
        print("No results to process")
    
    print("\nAnalysis complete!")
