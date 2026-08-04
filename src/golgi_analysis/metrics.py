import numpy as np
from numba import jit

@jit(nopython=True)
def get_distsq_to_center(img: np.ndarray) -> np.ndarray:
    """Calculate squared distances from each pixel to the weighted center of mass."""
    rows, cols = img.shape
    total_intensity = np.sum(img)
    if total_intensity == 0:
        return np.zeros_like(img)
    
    mean_row = 0.0
    mean_col = 0.0
    for i in range(rows):
        for j in range(cols):
            mean_row += img[i, j] * (i + 1)
            mean_col += img[i, j] * (j + 1)
            
    mean_row /= total_intensity
    mean_col /= total_intensity
    
    distsq = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row)**2 + ((j + 1) - mean_col)**2
            
    return distsq


@jit(nopython=True)
def get_distsq_to_centrosome_center(golgi_img: np.ndarray, centrosome_img: np.ndarray) -> np.ndarray:
    """Calculate squared distances from each pixel to the centrosome center of mass."""
    rows, cols = centrosome_img.shape
    total_intensity = np.sum(centrosome_img)
    
    if total_intensity == 0:
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
        
    distsq = np.zeros_like(golgi_img)
    for i in range(rows):
        for j in range(cols):
            distsq[i, j] = ((i + 1) - mean_row)**2 + ((j + 1) - mean_col)**2
            
    return distsq


@jit(nopython=True)
def get_radial_variance(img: np.ndarray, reference_img: np.ndarray = None) -> float:
    """Compute radial variance (weighted average of squared distances)."""
    if reference_img is not None:
        distsq_to_cm = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq_to_cm = get_distsq_to_center(img)
        
    total_intensity = np.sum(img)
    if total_intensity == 0:
        return 0.0
    return np.sum(img * distsq_to_cm) / total_intensity


@jit(nopython=True)
def get_radial_ecdf(img: np.ndarray, reference_img: np.ndarray = None) -> np.ndarray:
    """Create empirical cumulative distribution function of radial distances."""
    if reference_img is not None:
        distsq_to_cm = get_distsq_to_centrosome_center(img, reference_img)
    else:
        distsq_to_cm = get_distsq_to_center(img)
        
    rows, cols = img.shape
    max_dim = max(rows, cols)
    ecdf = np.zeros(max_dim)
    
    for r in range(1, max_dim + 1):
        r_squared = r * r
        cumulative_intensity = 0.0
        for i in range(rows):
            for j in range(cols):
                if distsq_to_cm[i, j] < r_squared:
                    cumulative_intensity += img[i, j]
        ecdf[r - 1] = cumulative_intensity
        
    max_val = np.max(ecdf)
    if max_val > 0:
        ecdf = ecdf / max_val
    return ecdf


def find_70pct_quantile(ecdf: np.ndarray) -> float:
    """Find the radius where ECDF reaches 0.7 (70%)."""
    if len(ecdf) == 0:
        return np.nan
    indices = np.where(ecdf >= 0.7)[0]
    if len(indices) == 0:
        return len(ecdf)
    return float(indices[0] + 1)