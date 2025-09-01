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


@jit(nopython=True)
def get_distsq_to_centrosome_center(golgi_img, centrosome_img):
    """Calculate squared distances from each pixel to the weighted center of mass of centrosome, 
    but return distances for the golgi image dimensions"""
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
                mean_row += centrosome_img[i, j] * (i + 1)  # R uses 1-based indexing
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
def get_radial_variance_from_centrosome(golgi_img, centrosome_img):
    """Compute radial variance of Golgi relative to centrosome center"""
    distsq_to_centrosome = get_distsq_to_centrosome_center(golgi_img, centrosome_img)
    total_intensity = np.sum(golgi_img)
    
    if total_intensity == 0:
        return 0.0
    
    return np.sum(golgi_img * distsq_to_centrosome) / total_intensity


@jit(nopython=True)
def get_radial_ecdf_from_centrosome(golgi_img, centrosome_img):
    """Create empirical cumulative distribution function of Golgi distances from centrosome center"""
    distsq_to_centrosome = get_distsq_to_centrosome_center(golgi_img, centrosome_img)
    rows, cols = golgi_img.shape
    max_dim = max(rows, cols)
    
    # Calculate ECDF for each radius
    ecdf = np.zeros(max_dim)
    
    for r in range(1, max_dim + 1):
        r_squared = r * r
        cumulative_intensity = 0.0
        
        for i in range(rows):
            for j in range(cols):
                if distsq_to_centrosome[i, j] < r_squared:
                    cumulative_intensity += golgi_img[i, j]
        
        ecdf[r-1] = cumulative_intensity
    
    # Normalize by maximum value
    max_val = np.max(ecdf)
    if max_val > 0:
        ecdf = ecdf / max_val
    
    return ecdf


def process_tiff_file(filename):
    """Process a single TIFF file to extract radial statistics for each segmented cell relative to centrosome"""
    print(f"Processing: {filename}")
    
    # Read TIFF file - read all pages separately
    full_path = os.path.join("imgs_segm", filename)
    print(f"  Reading file: {full_path}")
    
    # Read all pages from TIFF
    with tifffile.TiffFile(full_path) as tif:
        pages = [page.asarray() for page in tif.pages]
        print(f"  Found {len(pages)} pages")
        for i, page in enumerate(pages):
            print(f"    Page {i}: shape={page.shape}, dtype={page.dtype}")
    
    if len(pages) < 2:
        print(f"  Warning: Skipping {filename} - Expected at least 2 pages, got {len(pages)}")
        return pd.DataFrame()
    
    # Extract first page (microscopy data) - should have 3 channels
    first_img = pages[0]
    if first_img.ndim != 3 or first_img.shape[2] != 3:
        print(f"  Warning: Skipping {filename} - First page should have shape (H,W,3), got {first_img.shape}")
        return pd.DataFrame()
    
    # Extract last page (segmentation) - should have segmentation in first channel
    last_img = pages[-1]
    if last_img.ndim != 3:
        print(f"  Warning: Skipping {filename} - Last page should be 3D, got {last_img.ndim}D")
        return pd.DataFrame()
    
    # Extract channels
    img_golgi = first_img[:, :, 0].astype(np.float64)
    img_centr = first_img[:, :, 1].astype(np.float64) 
    img_dapi = first_img[:, :, 2].astype(np.float64)
    img_segm = last_img[:, :, 0].astype(np.float64)  # First channel of last page
    
    print(f"  Extracted channels - Golgi: {img_golgi.shape}, Centrosome: {img_centr.shape}, Segmentation: {img_segm.shape}")
    
    # Label segmented regions (equivalent to R's bwlabel)
    segm_binary = img_segm < 0.5
    segm_labeled, num_cells = label(segm_binary)
    
    print(f"  Found {num_cells} cells")
    
    # Calculate statistics for each cell
    statistics = []
    variances = []
    for cell_idx in range(1, num_cells + 1):
        # Create mask for current cell
        cell_mask = (segm_labeled == cell_idx).astype(np.float64)
        
        # Apply mask to both Golgi and centrosome images
        masked_golgi = img_golgi * cell_mask
        masked_centr = img_centr * cell_mask
        
        # Calculate radial ECDF and variance using centrosome center
        cell_ecdf = get_radial_ecdf_from_centrosome(masked_golgi, masked_centr)
        cell_variance = get_radial_variance_from_centrosome(masked_golgi, masked_centr)
        
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


def main():
    """Main processing pipeline for centrosome-centered analysis"""
    # Find all TIFF files
    tiff_files = []
    for root, dirs, files in os.walk("imgs_segm"):
        for file in files:
            if file.endswith('.tif'):
                rel_path = os.path.relpath(os.path.join(root, file), "imgs_segm")
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
        
        # Parse filename components (equivalent to R's separate function)
        combined_results['basename'] = combined_results['filename'].apply(
            lambda x: os.path.basename(x).replace('.tif', '')
        )
        
        # Split basename into components - handle spaces and multiple underscores (_+)
        def parse_basename(basename):
            # Replace spaces with underscores and split on one or more underscores
            parts = re.split(r'_+', basename.replace(' ', '_'))
            # Filter out empty parts
            parts = [part for part in parts if part]
            frame = parts[0] if len(parts) > 0 else ''
            cond = parts[1] if len(parts) > 1 else ''
            genotype = parts[2] if len(parts) > 2 else ''
            return pd.Series([frame, cond, genotype])
        
        combined_results[['frame', 'cond', 'genotype']] = combined_results['basename'].apply(parse_basename)
        
        return combined_results
    else:
        return pd.DataFrame()


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


def create_beeswarm_plots(results, suffix="centrosome"):
    """Create beeswarm plots for variance and 70% quantiles"""
    
    # Extract 70% quantiles
    quantiles_70pct = []
    for idx, row in results.iterrows():
        if len(row['statistic']) > 0:
            q70 = find_70pct_quantile(row['statistic'])
            if not np.isnan(q70):
                quantiles_70pct.append(q70)
            else:
                quantiles_70pct.append(np.nan)
        else:
            quantiles_70pct.append(np.nan)
    
    results['70pct_quantile'] = quantiles_70pct
    
    # Remove rows with NaN values
    plot_data = results.dropna(subset=['variance', '70pct_quantile'])
    
    print(f"Data for plotting: {len(plot_data)} cells")
    print(f"Conditions: {plot_data['cond'].value_counts().to_dict()}")
    
    # Plot 1: Standard deviation (sqrt of variance) beeswarm
    plt.figure(figsize=(10, 8))
    plot_data['std_dev'] = np.sqrt(plot_data['variance'])
    
    sns.swarmplot(data=plot_data, x='cond', y='std_dev', size=5, alpha=0.7)
    
    # Add mean lines
    for i, cond in enumerate(plot_data['cond'].unique()):
        subset = plot_data[plot_data['cond'] == cond]
        mean_val = subset['std_dev'].mean()
        plt.hlines(mean_val, i-0.25, i+0.25, colors='red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Condition')
    plt.ylabel('Standard Deviation (pixels)')
    plt.title(f'Radial Standard Deviation by Condition\n(Relative to Centrosome Center, Red lines show means)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'beeswarm_std_dev_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: 70% quantile beeswarm
    plt.figure(figsize=(10, 8))
    
    sns.swarmplot(data=plot_data, x='cond', y='70pct_quantile', size=5, alpha=0.7)
    
    # Add mean lines
    for i, cond in enumerate(plot_data['cond'].unique()):
        subset = plot_data[plot_data['cond'] == cond]
        mean_val = subset['70pct_quantile'].mean()
        plt.hlines(mean_val, i-0.25, i+0.25, colors='red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Condition')
    plt.ylabel('70% Quantile Radius (pixels)')
    plt.title(f'70% Quantile of Radial ECDF by Condition\n(Relative to Centrosome Center, Red lines show means)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'beeswarm_70pct_quantiles_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Beeswarm plots saved as beeswarm_std_dev_{suffix}.png and beeswarm_70pct_quantiles_{suffix}.png")
    
    # Print summary statistics
    print("\nStandard deviation summary:")
    print(plot_data.groupby('cond')['std_dev'].describe())
    print("\n70% quantile summary:")
    print(plot_data.groupby('cond')['70pct_quantile'].describe())


if __name__ == "__main__":
    # Run main processing
    results = main()
    
    if not results.empty:
        print(f"Processed {len(results)} cells from {results['filename'].nunique()} files")
        
        # Save results
        results.to_pickle("results_centrosome_center.pkl")
        print("Results saved to results_centrosome_center.pkl")
        
        # Create beeswarm plots
        create_beeswarm_plots(results, "centrosome")
        
    else:
        print("No results to process")