"""Cellpose segmentation and frame-to-frame tracking module."""

from pathlib import Path
import numpy as np
import tifffile
import torch
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from cellpose import models


def match_labels_to_previous(prev_mask, curr_mask, global_max_id, max_distance=100):
    new_mask = np.zeros_like(curr_mask)
    prev_ids = np.unique(prev_mask)[1:]
    curr_ids = np.unique(curr_mask)[1:]

    if len(prev_ids) == 0:
        for cid in curr_ids:
            global_max_id += 1
            new_mask[curr_mask == cid] = global_max_id
        return new_mask, global_max_id
    if len(curr_ids) == 0:
        return new_mask, global_max_id

    prev_centroids = ndimage.center_of_mass(prev_mask, prev_mask, prev_ids)
    curr_centroids = ndimage.center_of_mass(curr_mask, curr_mask, curr_ids)
    dist_matrix = distance.cdist(prev_centroids, curr_centroids)
    cost_matrix = dist_matrix.copy()
    cost_matrix[cost_matrix > max_distance] = 1e9

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_curr_indices = set()

    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < max_distance:
            new_mask[curr_mask == curr_ids[c]] = prev_ids[r]
            matched_curr_indices.add(c)

    next_new_id = global_max_id + 1
    for i, cid in enumerate(curr_ids):
        if i not in matched_curr_indices:
            new_mask[curr_mask == cid] = next_new_id
            next_new_id += 1

    return new_mask, max(global_max_id, int(np.max(new_mask)) if np.any(new_mask) else global_max_id)


def run_segmentation(
    illum_dir: str,
    mask_dir: str,
    nucleus_ch: int = 0,
    golgi_ch: int = 1,
    max_frames: int = 30
):
    print("\n" + "=" * 60)
    print("STEP 2 — Segmentation & Tracking (Cellpose)")
    print("=" * 60)

    input_dir, output_dir = Path(illum_dir), Path(mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.backends.mps.is_available()
    device = torch.device("mps") if use_gpu else torch.device("cpu")
    print(f"Hardware: {device.type.upper()} | Nucleus Ch: {nucleus_ch} | Golgi Ch: {golgi_ch}")
    model = models.CellposeModel(gpu=use_gpu, device=device)

    tif_files = sorted([f for f in input_dir.glob("*.tif") if "_masks" not in f.name])
    print(f"Found {len(tif_files)} files.\n")

    for file_path in tif_files:
        print(f"--- Processing: {file_path.name} ---")
        image_stack = tifffile.imread(str(file_path))[:max_frames]

        mask_list, last_valid_mask, last_valid_ref, global_max_id = [], None, None, 0
        list_for_cellpose = [np.moveaxis(frame[[nucleus_ch, golgi_ch]], 0, -1) for frame in image_stack]
        masks_raw, _, _ = model.eval(
            list_for_cellpose, batch_size=6, diameter=120, flow_threshold=0.4, cellprob_threshold=0.2
        )

        for t in range(len(image_stack)):
            curr_ref = image_stack[t, nucleus_ch]
            curr_mask = masks_raw[t]

            if np.mean(curr_ref) < 10.0:
                mask_list.append(np.zeros_like(curr_ref, dtype=np.uint16))
                continue

            if last_valid_mask is not None:
                shift_values, _, _ = phase_cross_correlation(last_valid_ref, curr_ref, upsample_factor=10)
                aligned_prev_mask = shift(last_valid_mask, -shift_values, order=0)
                curr_mask, global_max_id = match_labels_to_previous(
                    aligned_prev_mask, curr_mask, global_max_id, max_distance=100
                )
            else:
                global_max_id = int(np.max(curr_mask)) if np.any(curr_mask) else 0

            last_valid_mask, last_valid_ref = curr_mask.copy(), curr_ref.copy()
            mask_list.append(curr_mask.astype(np.uint16))

        output_filename = f"{file_path.stem}_masks.tif"
        tifffile.imwrite(
            str(output_dir / output_filename), np.stack(mask_list), imagej=True, metadata={"axes": "TYX"}
        )
        print(f"  > Saved: {output_filename} (Max ID: {global_max_id})")
        if use_gpu:
            torch.mps.empty_cache()

    print("\nStep 2 complete.")