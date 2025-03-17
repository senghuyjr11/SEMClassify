import os
import shutil
import random

# Define dataset paths
processed_root = "processed_dataset"  # Path to processed images
data_root = "processed_dataset_split"  # New path for train/val/test split
train_ratio = 0.8  # 80% training
val_ratio = 0.1  # 10% validation
test_ratio = 0.1  # 10% test

# Create train, validation, and test directories
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split images into train/val/test folders
for annealing_time in os.listdir(processed_root):
    annealing_path = os.path.join(processed_root, annealing_time)
    if os.path.isdir(annealing_path):
        for exposure_duration in os.listdir(annealing_path):
            exposure_path = os.path.join(annealing_path, exposure_duration)
            if os.path.isdir(exposure_path):
                images = os.listdir(exposure_path)
                random.shuffle(images)

                train_split = int(len(images) * train_ratio)
                val_split = int(len(images) * (train_ratio + val_ratio))

                train_images = images[:train_split]
                val_images = images[train_split:val_split]
                test_images = images[val_split:]

                train_category_path = os.path.join(train_dir, annealing_time, exposure_duration)
                val_category_path = os.path.join(val_dir, annealing_time, exposure_duration)
                test_category_path = os.path.join(test_dir, annealing_time, exposure_duration)
                os.makedirs(train_category_path, exist_ok=True)
                os.makedirs(val_category_path, exist_ok=True)
                os.makedirs(test_category_path, exist_ok=True)

                for img in train_images:
                    shutil.copy(os.path.join(exposure_path, img), os.path.join(train_category_path, img))
                for img in val_images:
                    shutil.copy(os.path.join(exposure_path, img), os.path.join(val_category_path, img))
                for img in test_images:
                    shutil.copy(os.path.join(exposure_path, img), os.path.join(test_category_path, img))

print(
    "Dataset split complete. Check 'processed_dataset_split/train', 'processed_dataset_split/val', and 'processed_dataset_split/test'.")
