from pathlib import Path
import numpy as np
import tifffile

# Set up dummy folder structure
base_dir = Path("./dummy_experiment")
raw_dir = base_dir / "rawImages"
mask_dir = base_dir / "filtered_masks"

raw_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

# Generate a fake 100x100 3-channel image [Nuc, Centrosome, Golgi]
fake_image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint16)

# Generate a fake mask with 2 labelled "cells"
fake_mask = np.zeros((100, 100), dtype=np.uint16)
fake_mask[20:40, 20:40] = 1  # Cell #1
fake_mask[60:80, 60:80] = 2  # Cell #2

# Save fake files with your standard naming structure
sample_names = [
    "Frame01_WT_nodrug_Exp1.tif",
    "Frame02_WT_2hdrug_Exp1.tif",
    "Frame03_CEP250KO_nodrug_Exp1.tif",
    "Frame04_CEP250KO_2hdrug_Exp1.tif",
]

for name in sample_names:
    tifffile.imwrite(raw_dir / name, fake_image)
    tifffile.imwrite(mask_dir / f"{Path(name).stem}_mask.tif", fake_mask)

print("✅ Fake test data successfully generated in ./dummy_experiment/")