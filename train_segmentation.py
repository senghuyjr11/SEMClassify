import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from corrosion_dataset import CorrosionSegmentationDataset
from unet import UNet
import os

# ✅ Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Use ResNet50 for segmentation
from torchvision import models
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True)  # ✅ Keep ResNet50
        self.base_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.base_model(x)['out']

# ✅ Reduce batch size to prevent out-of-memory errors (MX130 has limited VRAM)
BATCH_SIZE = 2  # Reduce batch size to 2 to fit MX130
GRAD_ACCUM_STEPS = 4  # ✅ Simulate larger batch size by accumulating gradients

# Set dataset paths
image_dir = "processed_dataset"
mask_dir = "processed_dataset_masks"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale images
])

# Load dataset
dataset = CorrosionSegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = UNet(num_classes=2).to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # ✅ Use AdamW for better stability
scaler = torch.cuda.amp.GradScaler()  # ✅ Enable Mixed Precision (FP16)

# ✅ Training loop with Mixed Precision & Gradient Accumulation
for epoch in range(20):
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast():  # ✅ Enable Mixed Precision for faster computation
            outputs = model(images)
            loss = criterion(outputs, masks.long()) / GRAD_ACCUM_STEPS  # ✅ Scale loss for accumulation

        scaler.scale(loss).backward()  # ✅ Scale gradient for FP16 training

        if (i + 1) % GRAD_ACCUM_STEPS == 0:  # ✅ Update weights only after accumulation steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

    pixel_accuracy = 100 * correct_pixels / total_pixels  # Compute pixel-wise accuracy

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.2f}%")

torch.save(model.state_dict(), "unet_corrosion.pth")
print("✅ Training complete. Model saved.")
