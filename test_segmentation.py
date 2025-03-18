import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from unet import UNet

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=2).to(device)
model.load_state_dict(torch.load("unet_corrosion.pth", map_location=device))
model.eval()

# Define transformation (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


image_path = "dataset/an_180/1hr/Img_Slide7_No5.png"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict segmentation mask
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.argmax(output, dim=1).cpu().squeeze().numpy()

# Overlay the predicted segmentation mask onto the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image, cmap="gray")
plt.imshow(pred_mask, alpha=0.5, cmap="jet")  # Overlay mask with transparency
plt.title("Predicted Segmentation")
plt.axis("off")

plt.show()