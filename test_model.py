import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define class labels (adjust based on your dataset)
class_labels = ["Low Corrosion", "Moderate Corrosion", "Severe Corrosion"]

# Load the trained fine-tuned model
model_path = "resnet50_finetuned.pth"
fine_tuned_model = models.resnet50(pretrained=False)
num_ftrs = fine_tuned_model.fc.in_features
fine_tuned_model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Ensure same dropout as training
    nn.Linear(num_ftrs, len(class_labels))
)
fine_tuned_model.load_state_dict(torch.load(model_path, map_location=device))
fine_tuned_model = fine_tuned_model.to(device)
fine_tuned_model.eval()

# Define image preprocessing (must match training preprocessing)
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Function to predict a single image
def predict_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = fine_tuned_model(image)
        _, predicted_class = output.max(1)

    predicted_label = class_labels[predicted_class.item()]

    # Display the image and prediction
    img = Image.open(image_path)
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

    print(f"Predicted Class: {predicted_label}")


# Function to generate Grad-CAM visualization
# Function to generate Grad-CAM visualization
def generate_gradcam(image_path, model, target_layer="layer4"):  # Updated target layer
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # Extract gradients and activations
    gradients = None
    activation = None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]  # Save gradients

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output  # Save activations

    # Register hooks
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    # Forward pass through the model
    model.zero_grad()
    outputs = model(image_tensor)
    _, predicted_class = outputs.max(1)

    # Backward pass to get gradients
    one_hot_output = torch.zeros(outputs.shape).to(device)
    one_hot_output[0][predicted_class] = 1
    outputs.backward(gradient=one_hot_output)

    # Ensure gradients and activations are captured
    if gradients is None or activation is None:
        print("Error: Gradients or activations not captured. Check the target layer name.")
        return

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activation.shape[1]):
        activation[:, i, :, :] *= pooled_gradients[i]
    # heatmap = torch.mean(activation, dim=1).squeeze().cpu().numpy()
    heatmap = torch.mean(activation, dim=1).squeeze().detach().cpu().numpy()

    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Convert heatmap to image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Show images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM: {class_labels[predicted_class.item()]}")
    plt.axis("off")

    plt.show()
    print(f"Predicted Class: {class_labels[predicted_class.item()]}")


if __name__ == "__main__":
    # Test the model with a sample image
    sample_image_path = "dataset/an_180/7days/Img_Slide12_No4.png"
    predict_image(sample_image_path)
    generate_gradcam(sample_image_path, fine_tuned_model)