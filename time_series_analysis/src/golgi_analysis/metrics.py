"""Numba-accelerated radial math and quantitative analysis methods."""

import os
import re
import numpy as np
import pandas as pd
import tifffile
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def get_distsq_to_center(img: np.ndarray) -> np.ndarray:
    rows, cols = img.shape
    total_intensity = np.sum(img)
    if total_intensity == 0:
        return np.zeros_like(img)

    mean_row = mean_col = 0.0
    for i in range(rows):
        for j in range(cols):
            mean_row += img[i, j] * (i + 1)
            mean_col += img[i, j] * (j + 1)
    mean_row /= total_intensity
    mean_col /= total_intensity

    distsq = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row) ** 2 + ((j + 1) - mean_col) ** 2
    return distsq


@jit(nopython=True, fastmath=True, cache=True)
def get_distsq_to_centrosome_center(golgi_img: np.ndarray, centrosome_img: np.ndarray) -> np.ndarray:
    rows, cols = centrosome_img.shape
    total_intensity = np.sum(centrosome_img)
    if total_intensity == 0:
        mean_row, mean_col = rows / 2.0, cols / 2.0
    else:
        mean_row = mean_col = 0.0
        for i in range(rows):
            for j in range(cols):
                mean_row += centrosome_img[i, j] * (i + 1)
                mean_col += centrosome_img[i, j] * (j + 1)
        mean_row /= total_intensity
        mean_col /= total_intensity

    distsq = np.zeros_like(golgi_img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row) ** 2 + ((j + 1) - mean_col) ** 2
    return distsq


@jit(nopython=True, fastmath=True, cache=True)
def get_radial_variance(img: np.ndarray, reference_img: np.ndarray = None) -> float:
    if reference_img is not None:
        distsq = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq = get_distsq_to_center(img)
    total = np.sum(img)
    return np.sum(img * distsq) / total if total != 0 else 0.0


@jit(nopython=True, fastmath=True, cache=True)
def get_radial_ecdf(img: np.ndarray, reference_img: np.ndarray = None) -> np.ndarray:
    if reference_img is not None:
        distsq = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq = get_distsq_to_center(img)
    rows, cols = img.shape
    max_dim = max(rows, cols)
    ecdf = np.zeros(max_dim)
    for r in range(1, max_dim + 1):
        r_sq = r * r
        cum = 0.0
        for i in range(rows):
            for j in range(cols):
                if distsq[i, j] < r_sq:
                    cum += img[i, j]
        ecdf[r - 1] = cum
    max_val = np.max(ecdf)
    if max_val > 0:
        ecdf = ecdf / max_val
    return ecdf


def find_70pct_quantile(ecdf: np.ndarray) -> float:
    if len(ecdf) == 0:
        return np.nan
    indices = np.where(ecdf >= 0.7)[0]
    return float(indices[0] + 1) if len(indices) > 0 else float(len(ecdf))


def parse_filename(basename: str) -> pd.Series:
    parts = [p for p in re.split(r"_+", basename) if p]
    if len(parts) >= 3:
        return pd.Series([parts[0], parts[1], parts[2]])
    return pd.Series(["unknown", "unknown", "unknown"])


def process_tiff_file(
    filename: str,
    illum_dir: str,
    filt_dir: str,
    analysis_mode: str = "centrosome",
    max_frames: int = 30,
    golgi_ch: int = 1,
    centrosome_ch: int = 2
) -> pd.DataFrame:
    """Computes radial metrics per cell across frames for a single TIFF image."""
    basename = os.path.basename(filename).replace(".tif", "")
    image_path = os.path.join(illum_dir, filename)
    mask_path = os.path.join(filt_dir, f"{basename}_masks_filtered.tif")

    image_data = tifffile.imread(image_path)
    mask_data = tifffile.imread(mask_path)

    if image_data.ndim == 3:
        image_data = np.expand_dims(image_data, axis=0)
        mask_data = np.expand_dims(mask_data, axis=0)

    image_data = image_data[:max_frames]
    mask_data = mask_data[:max_frames]

    num_frames = image_data.shape[0]
    statistics, variances, areas, cell_ids, frames = [], [], [], [], []

    for t in range(num_frames):
        img_golgi = image_data[t, golgi_ch, :, :].astype(np.float64)
        img_centr = image_data[t, centrosome_ch, :, :].astype(np.float64) if analysis_mode == "centrosome" else None
        frame_mask = mask_data[t].astype(np.float64)

        for cell_idx in np.unique(frame_mask)[1:]:
            cell_mask = (frame_mask == cell_idx).astype(np.float64)
            cell_area = np.sum(cell_mask)

            masked_golgi = img_golgi * cell_mask

            if analysis_mode == "centrosome":
                masked_centr = img_centr * cell_mask
                cell_ecdf = get_radial_ecdf(masked_golgi, masked_centr)
                cell_variance = get_radial_variance(masked_golgi, masked_centr)
            else:
                cell_ecdf = get_radial_ecdf(masked_golgi)
                cell_variance = get_radial_variance(masked_golgi)

            hit = np.where(cell_ecdf >= 0.9999)[0]
            if len(hit) > 0:
                cell_ecdf = cell_ecdf[: hit[0] + 1]

            statistics.append(cell_ecdf)
            variances.append(cell_variance)
            areas.append(cell_area)
            cell_ids.append(cell_idx)
            frames.append(t)

    df = pd.DataFrame({
        "filename": [filename] * len(statistics),
        "frame": frames,
        "cell_idx": cell_ids,
        "statistic": statistics,
        "variance": variances,
        "cell_area": areas,
    })
    return df