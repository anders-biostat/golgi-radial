#!/usr/bin/env python3
"""
Image Annotation Script
Converts R annotation code to Python for labeling segmented cells in microscopy images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tifffile
from scipy.ndimage import label, binary_erosion
import matplotlib.pyplot as plt
import os
import re
from glob import glob


# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
INPUT_DIR = "imgs_segm"
OUTPUT_DIR = "imgs_anno"

# Annotation settings
ANNOTATION_COLOR = (255, 255, 255)  # White text
FONT_SIZE = 15  # Text size for cell labels
BOUNDARY_THICKNESS = 2  # Cell boundary line thickness

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# ANNOTATION FUNCTIONS
# =============================================================================

def create_cell_boundaries(segm_binary):
    """Create cell boundary annotations by subtracting eroded version"""
    # Erode the segmentation mask
    eroded = binary_erosion(segm_binary, iterations=1)
    
    # Boundary is original minus eroded (equivalent to R's (img_segm<.1) - erode(img_segm<.1))
    boundaries = segm_binary.astype(np.float64) - eroded.astype(np.float64)
    
    # Scale to 0.5 as in R code (divided by 2)
    boundaries = boundaries / 2.0
    
    return boundaries


def create_text_image(text, font_size=FONT_SIZE):
    """Create a text image similar to R's sprintf + image_read + image_convert"""
    # Try to use a system font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Create a temporary image to measure text size
    temp_img = Image.new('L', (200, 50), color=0)
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    
    # Handle negative top values and ensure full character height
    left, top, right, bottom = bbox
    text_width = right - left
    text_height = bottom - top
    
    # Add extra padding to ensure characters don't get clipped
    padding = 4
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding
    
    # Create the actual text image with proper size
    text_img = Image.new('L', (img_width, img_height), color=0)  # Black background
    draw = ImageDraw.Draw(text_img)
    
    # Position text accounting for any negative top offset
    text_x = padding - left
    text_y = padding - top
    draw.text((text_x, text_y), text, fill=255, font=font)  # White text
    
    # Convert to numpy array (equivalent to R's image_data() %>% as.integer() %>% drop() %>% t())
    text_array = np.array(text_img).astype(np.float64)
    
    return text_array


def place_text_on_image(anno_image, text_array, center_row, center_col):
    """Place text image on annotation image at specified center position"""
    rows, cols = anno_image.shape
    text_rows, text_cols = text_array.shape
    
    # Calculate start position (center the text)
    start_row = int(round(center_row - text_rows / 2))
    start_col = int(round(center_col - text_cols / 2))
    
    # Ensure we don't go out of bounds - clamp positions to valid ranges
    start_row = max(0, min(start_row, rows - text_rows))
    start_col = max(0, min(start_col, cols - text_cols))
    
    # Calculate end positions
    end_row = start_row + text_rows
    end_col = start_col + text_cols
    
    # Double check bounds (shouldn't be necessary with proper clamping above)
    end_row = min(end_row, rows)
    end_col = min(end_col, cols)
    
    # Only proceed if we have valid dimensions
    if start_row < rows and start_col < cols and end_row > start_row and end_col > start_col:
        # Calculate actual region dimensions
        actual_text_rows = end_row - start_row
        actual_text_cols = end_col - start_col
        
        # Extract regions - make sure they match in size
        region = anno_image[start_row:end_row, start_col:end_col]
        text_region = text_array[:actual_text_rows, :actual_text_cols]
        
        # Apply the text overlay (equivalent to R's blending formula)
        # R code: 1 - ( 1 - anno[region] ) * m/255
        # This creates a "lighten" blend mode
        anno_image[start_row:end_row, start_col:end_col] = (
            1 - (1 - region) * (1 - text_region / 255)
        )
    
    return anno_image


def annotate_image(filename):
    """Process a single TIFF file to create annotated version"""
    print(f"Annotating: {filename}")
    
    # Read TIFF file
    full_path = os.path.join(INPUT_DIR, filename)
    
    try:
        with tifffile.TiffFile(full_path) as tif:
            pages = [page.asarray() for page in tif.pages]
    except Exception as e:
        print(f"  Error reading {filename}: {e}")
        return False
    
    if len(pages) < 2:
        print(f"  Warning: Skipping {filename} - Expected at least 2 pages, got {len(pages)}")
        return False
    
    # Extract images
    img_anno = pages[0].astype(np.float64) / 255.0  # Normalize to 0-1 range
    img_segm = pages[-1][:, :, 0].astype(np.float64) / 255.0  # First channel of last page
    
    print(f"  Image shape: {img_anno.shape}, Segmentation shape: {img_segm.shape}")
    
    # Create binary segmentation (same threshold as analysis script for consistency)
    segm_binary = img_segm < 0.5
    
    # Create cell boundaries
    anno = create_cell_boundaries(segm_binary)
    
    # Label connected components
    segm_labeled, num_cells = label(segm_binary)
    print(f"  Found {num_cells} cells")
    
    # Add text labels for each cell
    for cell_idx in range(1, num_cells + 1):
        # Find center of mass for this cell (equivalent to R's weighted mean calculation)
        cell_mask = (segm_labeled == cell_idx)
        
        if np.sum(cell_mask) == 0:
            continue
        
        # Calculate weighted center of mass using R's approach:
        # mean_row <- sum( (segm_labeled==cell) * row(anno) ) / sum(segm_labeled==cell)
        # mean_col <- sum( (segm_labeled==cell) * col(anno) ) / sum(segm_labeled==cell)
        
        # Create row and column index arrays (1-based like R)
        row_indices, col_indices = np.meshgrid(np.arange(1, anno.shape[0] + 1), 
                                               np.arange(1, anno.shape[1] + 1), 
                                               indexing='ij')
        
        # Calculate weighted centers using the anno image values
        total_weight = np.sum(cell_mask)
        mean_row = np.sum(cell_mask * row_indices) / total_weight
        mean_col = np.sum(cell_mask * col_indices) / total_weight
        
        # Create text image for this cell
        text = f"{cell_idx}"
        text_array = create_text_image(text)
        
        # Place text on annotation image
        anno = place_text_on_image(anno, text_array, mean_row, mean_col)
    
    # Apply annotation overlay to original image (R's blending formula)
    # R code: img_anno[,,1] <- 1 - (1-img_anno[,,1]) * (1-anno)
    if img_anno.ndim == 3:
        # Apply to first two channels (typically red and green)
        for channel in [0, 1]:
            img_anno[:, :, channel] = 1 - (1 - img_anno[:, :, channel]) * (1 - anno)
    else:
        # Grayscale image
        img_anno = 1 - (1 - img_anno) * (1 - anno)
    
    # Convert back to 0-255 range
    img_anno = np.clip(img_anno * 255, 0, 255).astype(np.uint8)
    
    # Create output filename (equivalent to R's string manipulation)
    basename = os.path.basename(filename).replace('.tif', '')
    output_filename = f"{basename}.anno.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the annotated image
    if img_anno.ndim == 3:
        # Color image
        Image.fromarray(img_anno, mode='RGB').save(output_path)
    else:
        # Grayscale image
        Image.fromarray(img_anno, mode='L').save(output_path)
    
    print(f"  Saved: {output_path}")
    return True


def annotate_all_images():
    """Process all TIFF files in the input directory"""
    print("=" * 60)
    print("IMAGE ANNOTATION")
    print("=" * 60)
    
    # Find all TIFF files (equivalent to R's list.files with pattern and recursive)
    tiff_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.tif'):
                rel_path = os.path.relpath(os.path.join(root, file), INPUT_DIR)
                tiff_files.append(rel_path)
    
    print(f"Found {len(tiff_files)} TIFF files to annotate")
    
    success_count = 0
    for i, filename in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] Processing {filename}")
        try:
            if annotate_image(filename):
                success_count += 1
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"ANNOTATION COMPLETE")
    print(f"Successfully annotated {success_count}/{len(tiff_files)} files")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    annotate_all_images()