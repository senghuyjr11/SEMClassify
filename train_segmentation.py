import torch.multiprocessing
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from corrosion_dataset import CorrosionSegmentationDataset
from unet import UNet

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # ✅ Must be the first line

    # ✅ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Reduce num_workers to prevent crashing
    BATCH_SIZE = 16
    NUM_WORKERS = 2  # ✅ Reduce for stability (or set to 0 if still crashing)
    GRAD_ACCUM_STEPS = 2  # ✅ Gradient accumulation

    # Set dataset paths
    image_dir = "processed_dataset"
    mask_dir = "processed_dataset_masks"

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale images
    ])

    # ✅ Load dataset inside `if __name__ == "__main__"`
    dataset = CorrosionSegmentationDataset(image_dir, mask_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # ✅ Initialize model inside `if __name__ == "__main__"`
    model = UNet(num_classes=2).to(device)

    # ✅ Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # ✅ Mixed Precision (FP16)

    # ✅ Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            with torch.cuda.amp.autocast():  # ✅ Enable FP16
                outputs = model(images)
                loss = criterion(outputs, masks.long()) / GRAD_ACCUM_STEPS  # ✅ Gradient accumulation

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:  # ✅ Update weights after accumulation steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

        pixel_accuracy = 100 * correct_pixels / total_pixels
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.2f}%")

    torch.save(model.state_dict(), "unet_corrosion.pth")
    print("✅ Training complete. Model saved.")