import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tifffile
from scipy.ndimage import label
from numba import jit
import matplotlib.pyplot as plt
from glob import glob
import os


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
def get_radial_variance(img):
    """Compute radial variance (weighted average of squared distances)"""
    distsq_to_cm = get_distsq_to_center(img)
    total_intensity = np.sum(img)
    
    if total_intensity == 0:
        return 0.0
    
    return np.sum(img * distsq_to_cm) / total_intensity


@jit(nopython=True)
def get_radial_ecdf(img):
    """Create empirical cumulative distribution function of radial distances"""
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


def process_tiff_file(filename):
    """Process a single TIFF file to extract radial statistics for each segmented cell"""
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
    
    print(f"  Extracted channels - Golgi: {img_golgi.shape}, Segmentation: {img_segm.shape}")
    
    # Label segmented regions (equivalent to R's bwlabel)
    segm_binary = img_segm < 0.5
    segm_labeled, num_cells = label(segm_binary)
    
    # Calculate statistics for each cell
    statistics = []
    for cell_idx in range(1, num_cells + 1):
        # Create mask for current cell
        cell_mask = (segm_labeled == cell_idx).astype(np.float64)
        
        # Apply mask to Golgi image
        masked_golgi = img_golgi.astype(np.float64) * cell_mask
        
        # Calculate radial ECDF
        cell_ecdf = get_radial_ecdf(masked_golgi)
        statistics.append(cell_ecdf)
    
    # Create results dataframe
    results = pd.DataFrame({
        'filename': [filename] * len(statistics),
        'cell_idx': range(1, len(statistics) + 1),
        'statistic': statistics
    })
    
    return results


def main():
    """Main processing pipeline"""
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
            import re
            parts = re.split(r'_+', basename.replace(' ', '_'))
            # Filter out empty parts
            parts = [part for part in parts if part]
            frame = parts[0] if len(parts) > 0 else ''
            cond = parts[1] if len(parts) > 1 else ''
            genotype = parts[2] if len(parts) > 2 else ''
            return pd.Series([frame, cond, genotype])
        
        combined_results[['frame', 'cond', 'genotype']] = combined_results['basename'].apply(parse_basename)
        
        # Calculate variance for each statistic (for plotting)
        combined_results['variance'] = combined_results['statistic'].apply(
            lambda x: np.var(x) if len(x) > 0 else 0
        )
        
        return combined_results
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Run main processing
    results = main()
    
    if not results.empty:
        print(f"Processed {len(results)} cells from {results['filename'].nunique()} files")
        print("\nFirst few results:")
        print(results.head())
        
        # Save results
        results.to_pickle("results.pkl")
        print("\nResults saved to results.pkl")
        
        # Basic plotting (equivalent to R's ggplot) with horizontal jitter
        if 'cond' in results.columns and 'variance' in results.columns:
            plt.figure(figsize=(10, 6))
            conditions = results['cond'].unique()
            np.random.seed(42)  # For reproducible jitter
            
            for i, cond in enumerate(conditions):
                subset = results[results['cond'] == cond]
                # Add horizontal jitter
                jitter = np.random.normal(0, 0.05, len(subset))
                x_positions = i + jitter
                plt.scatter(x_positions, np.sqrt(subset['variance']), 
                           alpha=0.6, label=cond, s=30)
            
            plt.xticks(range(len(conditions)), conditions)
            plt.ylabel('sqrt(variance)')
            plt.xlabel('Condition')
            plt.title('Radial Variance by Condition')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('radial_variance_plot.png', dpi=150, bbox_inches='tight')
            plt.close()  # Close the plot instead of showing it
            print("Plot saved as radial_variance_plot.png")
    else:
        print("No results to process")