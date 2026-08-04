from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile

from .metrics import (
    get_radial_ecdf,
    get_radial_variance,
    find_70pct_quantile
)

def parse_filename(basename: str) -> pd.Series:
    """Parse filename components: Frame_Genotype_Condition_Experiment."""
    parts = [part for part in re.split(r'_+', basename) if part]
    if len(parts) >= 4:
        frame, genotype, condition, experiment = parts[:4]
    elif len(parts) == 3:
        frame, genotype, condition = parts
        experiment = 'unknown'
    else:
        return pd.Series(['unknown', 'unknown', 'unknown', 'unknown'])
    
    return pd.Series([frame, genotype, condition, experiment])


def process_tiff_file(
    image_path: Path, 
    mask_path: Path, 
    analysis_mode: str = 'golgi'
) -> pd.DataFrame:
    """Extract radial metrics for each cell in a multi-channel TIFF stack."""
    if not mask_path.exists():
        print(f"Warning: Mask not found for {image_path.name}, skipping.")
        return pd.DataFrame()

    image_data = tifffile.imread(str(image_path))
    mask_data = tifffile.imread(str(mask_path))

    # Channels: 0: Nuclei, 1: Centrosome, 2: Golgi
    img_centr = image_data[:, :, 1].astype(np.float64)
    img_golgi = image_data[:, :, 2].astype(np.float64)
    img_segm = mask_data.astype(np.float64)

    # Calculate background subtraction levels
    background_mask = (img_segm == 0)
    if np.sum(background_mask) > 0:
        golgi_bg = np.median(img_golgi[background_mask])
        centr_bg = np.median(img_centr[background_mask])
    else:
        golgi_bg, centr_bg = 0.0, 0.0

    img_golgi_corr = np.maximum(img_golgi - golgi_bg, 0)
    img_centr_corr = np.maximum(img_centr - centr_bg, 0)

    num_cells = int(np.max(img_segm))
    statistics, variances, areas = [], [], []

    for cell_idx in range(1, num_cells + 1):
        cell_mask = (img_segm == cell_idx).astype(np.float64)
        areas.append(np.sum(cell_mask))

        masked_golgi = img_golgi_corr * cell_mask
        masked_centr = img_centr_corr * cell_mask

        if analysis_mode == 'centrosome':
            cell_ecdf = get_radial_ecdf(masked_golgi, masked_centr)
            cell_var = get_radial_variance(masked_golgi, masked_centr)
        else:
            cell_ecdf = get_radial_ecdf(masked_golgi)
            cell_var = get_radial_variance(masked_golgi)

        statistics.append(cell_ecdf)
        variances.append(cell_var)

    return pd.DataFrame({
        'filename': [image_path.name] * len(statistics),
        'cell_idx': range(1, len(statistics) + 1),
        'statistic': statistics,
        'variance': variances,
        'cell_area': areas
    })


def run_analysis_pipeline(
    input_dir: Path, 
    mask_dir: Path, 
    analysis_mode: str = 'golgi'
) -> pd.DataFrame:
    """Run image batch analysis and build combined results DataFrame."""
    tiff_files = sorted(list(input_dir.glob("*.tif*")))
    print(f"Processing {len(tiff_files)} TIFF files in {analysis_mode} mode...")

    all_results = []
    for img_path in tiff_files:
        mask_path = mask_dir / f"{img_path.stem}_mask.tif"
        try:
            res = process_tiff_file(img_path, mask_path, analysis_mode=analysis_mode)
            if not res.empty:
                all_results.append(res)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    if not all_results:
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    df['basename'] = df['filename'].apply(lambda x: Path(x).stem)
    df[['frame', 'genotype', 'cond', 'experiment']] = df['basename'].apply(parse_filename)
    
    # Calculate 70% quantiles
    df['70pct_quantile'] = df['statistic'].apply(
        lambda ecdf: find_70pct_quantile(ecdf) if len(ecdf) > 0 else np.nan
    )
    return df