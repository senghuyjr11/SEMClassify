import cv2
import numpy as np
import os

# Use your existing dataset
image_dir = "processed_dataset"  # Use this instead of segmentation_dataset/images
mask_dir = "processed_dataset_masks"  # Save masks separately
os.makedirs(mask_dir, exist_ok=True)

# Iterate over subfolders (AN_30, AN_60, etc.)
for category in os.listdir(image_dir):
    category_path = os.path.join(image_dir, category)
    mask_category_path = os.path.join(mask_dir, category)
    os.makedirs(mask_category_path, exist_ok=True)

    # Iterate over subfolders (1hr, 7days, etc.)
    for timepoint in os.listdir(category_path):
        timepoint_path = os.path.join(category_path, timepoint)
        mask_timepoint_path = os.path.join(mask_category_path, timepoint)
        os.makedirs(mask_timepoint_path, exist_ok=True)

        for filename in os.listdir(timepoint_path):
            image_path = os.path.join(timepoint_path, filename)
            mask_path = os.path.join(mask_timepoint_path, filename)

            # Read image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply adaptive thresholding to highlight corroded areas
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply Morphological Operations to refine mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

            # Save the generated mask
            cv2.imwrite(mask_path, mask)

print("✅ Auto-generated segmentation masks saved in:", mask_dir)
