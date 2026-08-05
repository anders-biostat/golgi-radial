"""BaSiC Illumination Correction module."""

import os
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from basicpy import BaSiC


def run_illumination_correction(
    raw_dir: str,
    illum_dir: str,
    qc_dir: str,
    max_frames: int = 30,
    get_darkfield: bool = False,
    smoothness: float = 1.0,
    percentile_clip: float = 99.9
):
    print("\n" + "=" * 60)
    print("STEP 1 — Illumination Correction (BaSiC)")
    print("=" * 60)

    input_folder, output_folder, qc_folder = Path(raw_dir), Path(illum_dir), Path(qc_dir)
    output_folder.mkdir(exist_ok=True, parents=True)
    qc_folder.mkdir(exist_ok=True, parents=True)

    tif_files = [f for f in input_folder.glob("*.tif*") if "_corrected" not in f.name]
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_folder}")
    print(f"Found {len(tif_files)} file(s) to process.")

    probe = tifffile.imread(str(tif_files[0]))[:max_frames]
    n_frames, n_channels, n_y, n_x = probe.shape
    original_dtype = probe.dtype
    print(f"Shape: {n_frames} frames × {n_channels} ch × {n_y} × {n_x} | dtype: {original_dtype}")

    print("\n--- Fitting flatfields (pooling all files) ---")
    basics = {}

    for c in range(n_channels):
        print(f"  Channel {c}: loading frames...", end=" ")
        all_frames = []
        for fpath in tif_files:
            stack = tifffile.imread(str(fpath))[:max_frames]
            all_frames.append(stack[:, c, :, :])
        training_data = np.concatenate(all_frames, axis=0)
        print(f"{training_data.shape[0]} frames total.")

        basic = BaSiC(get_darkfield=get_darkfield, smoothness_flatfield=smoothness)
        basic.fit(training_data)
        basics[c] = basic

        ff = basic.flatfield
        fig, axes = plt.subplots(1, 2 if get_darkfield else 1, figsize=(8 if get_darkfield else 4, 3))
        if not get_darkfield:
            axes = [axes]
        axes[0].imshow(ff, cmap="gray")
        axes[0].set_title(f"Flatfield Ch{c} (min={ff.min():.3f} max={ff.max():.3f})")
        axes[0].axis("off")
        if get_darkfield:
            axes[1].imshow(basic.darkfield, cmap="gray")
            axes[1].set_title(f"Darkfield Ch{c}")
            axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(qc_folder / f"flatfield_ch{c}.png", dpi=150)
        plt.close()

    print("\n--- Applying corrections ---")
    dtype_max = np.iinfo(original_dtype).max

    for file_path in tif_files:
        print(f"  {file_path.name}")
        try:
            image_stack = tifffile.imread(str(file_path))[:max_frames]
            corrected_stack = np.zeros_like(image_stack, dtype=np.float32)

            for c in range(n_channels):
                channel_data = image_stack[:, c, :, :]
                corrected = basic.transform(channel_data)
                corrected_stack[:, c, :, :] = np.clip(corrected, 0, None)

            final = np.zeros_like(corrected_stack)
            for c in range(n_channels):
                ch = corrected_stack[:, c, :, :]
                p_high = np.percentile(ch, percentile_clip)
                final[:, c, :, :] = ch / p_high * dtype_max * 0.98 if p_high > 0 else ch

            final = np.clip(final, 0, dtype_max).astype(original_dtype)
            save_name = f"{file_path.stem}_corrected.tif"
            tifffile.imwrite(output_folder / save_name, final, imagej=True, metadata={"axes": "TCYX"})
            print(f"    Saved → {save_name}")

        except Exception as e:
            print(f"    ! Error processing {file_path.name}: {e}")

    print("\nStep 1 complete.")