import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob  # ✅ Allows searching for images in subdirectories

class CorrosionSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # ✅ Recursively find images
        self.images = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)) + \
                      sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)) + \
                      sorted(glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True))

        if len(self.images) == 0:
            raise ValueError(f"❌ No images found in {image_dir}. Check dataset structure!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = image_path.replace(self.image_dir, self.mask_dir)  # Ensure masks match images

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # ✅ Convert mask to tensor

        # ✅ Ensure mask values are {0, 1} instead of {0, 255}
        mask = (mask > 0).long().squeeze(0)  # Convert to binary {0, 1} and remove channel

        return image, mask
