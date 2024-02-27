import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification


def weighted_feature_fusion(original_feature, vit_feature, alpha=0.7, beta=0.3):
    """
    Reads a PNG image, extracts features, and performs weighted feature fusion.

    Args:
      img_path: Path to the PNG image file.
      vit_feature: A 3-dimensional numpy array representing the ViT feature (height, width, channels).
      original_feature: A 3-dimensional numpy array representing the original feature (height, width, channels).
      alpha: Weight for the ViT feature (between 0 and 1).
      beta: Weight for the original feature (between 0 and 1).

    Returns:
      A 3-dimensional numpy array representing the combined feature (height, width, channels).
    """

    # Ensure shape compatibility with features
    if img_array.shape[0:2] != vit_feature.shape[0:2] or img_array.shape[2] != 3:
        raise ValueError("Image size and feature shapes must be compatible.")

    # Perform weighted feature fusion
    combined_feature = (alpha * vit_feature + beta * original_feature).astype("uint8")

    return combined_feature


# Example feature extraction (replace with your actual implementation)
def extract_original_features(img_array):
    # Implement your feature extraction method here, returning a 3D array
    # with the same dimensions as the image (height, width, channels)
    # Example: extract mean RGB values
    return np.mean(img_array, axis=(0, 1))


def extract_vit_features(image):
    # Load the ViT feature extractor
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs.pixel_values


# Example usage with placeholder data
img_path = "/Users/ducthong/Desktop/AI/Computer Vision/Vision Transformer/dataset/Plant_leaf_diseases_dataset/train/Apple___Apple_scab/image (2).JPG"  # Replace with your image path

# Read image and convert to RGB array
img = Image.open(img_path).convert("RGB")
img_array = np.array(img.resize((224, 224)))
vit_feature = extract_vit_features(img)
# Define transformations
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert tensor to PIL image
        transforms.ToTensor(),  # Convert PIL image to tensor again
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # Normalize for visualization
        transforms.ToPILImage(),  # Convert back to PIL image
    ]
)

# Convert tensor to PIL image
image_proccessed = np.array(transform(vit_feature[0]))
combined_feature = weighted_feature_fusion(
    img_array, image_proccessed, alpha=0.7, beta=0.3
)

# Display the combined feature image
img_combined = Image.fromarray(combined_feature)
img_combined.show()  # Displays the image using PIL
