import os

from PIL import Image
from torchvision import transforms

# Define input and output paths
dataset_root = "dataset"
processed_root = "processed_dataset"

# Ensure processed dataset folder exists
os.makedirs(processed_root, exist_ok=True)

# Define transformations (Resizing, Normalization, Augmentation)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((224, 224)),  # Resize to 224x224 for CNN models
    transforms.RandomHorizontalFlip(),  # Augment with horizontal flip
    transforms.RandomRotation(15),  # Augment with random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
])

# Process images category-wise
for annealing_time in os.listdir(dataset_root):
    annealing_path = os.path.join(dataset_root, annealing_time)
    if os.path.isdir(annealing_path):
        for exposure_duration in os.listdir(annealing_path):
            exposure_path = os.path.join(annealing_path, exposure_duration)
            processed_folder = os.path.join(processed_root, annealing_time, exposure_duration)
            os.makedirs(processed_folder, exist_ok=True)

            for img_name in os.listdir(exposure_path):
                img_path = os.path.join(exposure_path, img_name)
                try:
                    img = Image.open(img_path)
                    img = transform(img)  # Apply preprocessing transformations

                    # Save processed image
                    processed_img_path = os.path.join(processed_folder, f"Processed_{img_name}")
                    transforms.ToPILImage()(img).save(processed_img_path)
                    print(f"Processed and saved: {processed_img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

print("Processing complete. Check the processed_dataset folder.")
