import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from corrosion_dataset import CorrosionSegmentationDataset
from unet import UNet
import os

# Set paths
image_dir = "processed_dataset"
mask_dir = "processed_dataset_masks"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
])

# Load dataset
dataset = CorrosionSegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=2).to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(20):  # Adjust as needed
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.long())  # Masks must be long dtype for CE Loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "unet_corrosion.pth")
print("✅ Training complete. Model saved.")
