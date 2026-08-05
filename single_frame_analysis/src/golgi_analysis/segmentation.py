from pathlib import Path
import numpy as np
import torch
from skimage.io import imread, imsave
from skimage.segmentation import clear_border
from skimage import measure
from cellpose import models

def get_compute_device() -> torch.device:
    """Detect GPU, Apple MPS, or CPU acceleration."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_cellpose_segmentation(
    image_dir: Path, 
    raw_mask_dir: Path, 
    model_path: str, 
    diameter: int = 50, 
    batch_size: int = 3
):
    """Run batch Cellpose segmentation on images."""
    raw_mask_dir.mkdir(parents=True, exist_ok=True)
    device = get_compute_device()
    print(f"Using compute device: {device}")

    model = models.CellposeModel(pretrained_model=model_path, device=device)
    image_files = sorted([f for f in image_dir.glob("*.tif")])
    
    if not image_files:
        raise FileNotFoundError(f"No .tif images found in {image_dir}")

    print(f"Found {len(image_files)} images for segmentation.")

    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        imgs = [imread(str(p)) for p in batch_paths]
        
        masks, _, _ = model.eval(
            imgs,
            channels=[0, 0],
            diameter=diameter,
            do_3D=False,
            batch_size=batch_size
        )

        for path, mask in zip(batch_paths, masks):
            save_path = raw_mask_dir / f"{path.stem}_mask.tif"
            imsave(str(save_path), mask.astype(np.uint16), check_contrast=False)


def filter_masks(raw_mask_dir: Path, filtered_mask_dir: Path, min_size: int = 500):
    """Remove border-touching cells and filter out objects smaller than min_size."""
    filtered_mask_dir.mkdir(parents=True, exist_ok=True)
    mask_files = list(raw_mask_dir.glob("*.tif*"))

    print(f"Filtering {len(mask_files)} masks...")
    for path in mask_files:
        mask = imread(str(path)).astype(np.int32)
        
        # 1. Clear image borders
        mask_cleared = clear_border(mask)
        
        # 2. Filter by minimum area
        labels, counts = np.unique(mask_cleared, return_counts=True)
        valid_mask = (counts >= min_size) & (labels != 0)
        valid_labels = labels[valid_mask]
        
        mask_filtered = np.where(np.isin(mask_cleared, valid_labels), mask_cleared, 0)
        
        # 3. Renumber labels sequentially
        mask_final = measure.label(mask_filtered)
        
        save_path = filtered_mask_dir / path.name
        imsave(str(save_path), mask_final.astype(np.uint16), check_contrast=False)