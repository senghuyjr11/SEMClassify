import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from torch.utils.data import DataLoader
from corrosion_dataset import CorrosionSegmentationDataset
from torchvision import transforms
from unet import UNet
import matplotlib.pyplot as plt

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=2).to(device)
model.load_state_dict(torch.load("unet_corrosion.pth", map_location=device))
model.eval()

# Define dataset paths
image_dir = "processed_dataset"
mask_dir = "processed_dataset_masks"

# Define transformations (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load dataset
dataset = CorrosionSegmentationDataset(image_dir, mask_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

def evaluate_segmentation(model, dataloader):
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for pred, gt in zip(preds, masks):
                iou = jaccard_score(gt.flatten(), pred.flatten(), average='macro')
                dice = f1_score(gt.flatten(), pred.flatten(), average='macro')

                iou_scores.append(iou)
                dice_scores.append(dice)

    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")

# Sample visualization function
def visualize_predictions(model, dataloader, num_samples=5):
    model.eval()
    images, masks = next(iter(dataloader))
    # images, masks = images[:num_samples], masks[:num_samples]
    num_samples = min(num_samples, images.shape[0])  # ✅ Ensure num_samples ≤ batch size
    images, masks = images[:num_samples], masks[:num_samples]

    with torch.no_grad():
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        # axes[i, 0].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[i, 0].imshow(images[i].cpu().squeeze().permute(1, 2, 0), cmap="gray")
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i], cmap="jet")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.show()

if __name__ == "__main__":
    evaluate_segmentation(model, test_loader)
    visualize_predictions(model, test_loader)
