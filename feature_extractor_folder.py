import torch
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import time
import numpy as np
import os

# Load the ViT image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


def extract_vit_features(image):
    # Load the ViT feature extractor
    inputs = processor(images=image, return_tensors="pt")
    return inputs.pixel_values


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


def transform_image(features):
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
    return np.array(transform(features[0]))


def read_folder_names(folder_path):
    """Reads all folder names within a given folder path."""
    folder_names = []
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):
            folder_names.append(item)
    return folder_names


def remove_files_in_folder(folder_path):
    """Removes all files within a given folder path, but not the folder itself."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


ROOT_FOLDER = "/Users/ducthong/Desktop/AI/Computer Vision/Vision Transformer/dataset/Plant_leaf_diseases_dataset/"
DESTINATION_FOLDER = "/Users/ducthong/Desktop/AI/Computer Vision/Vision Transformer/features_extracted_dataset/"
if not os.path.exists(DESTINATION_FOLDER):
    os.makedirs(DESTINATION_FOLDER)
subfolder_names = read_folder_names(ROOT_FOLDER)
all_classes_names = read_folder_names(f"{ROOT_FOLDER}/{subfolder_names[0]}")

print(subfolder_names)
print(all_classes_names)

for subfolder_name in subfolder_names:
    for class_name in all_classes_names:
        print(class_name)
        image_folder = f"{ROOT_FOLDER}/{subfolder_name}/{class_name}"
        output_folder = f"{DESTINATION_FOLDER}{subfolder_name}/{class_name}"
        print(output_folder)
        # Create the destination folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img.resize((224, 224)))
            vit_feature = extract_vit_features(img)
            vit_feature_transformed = transform_image(vit_feature)
            combined_feature = weighted_feature_fusion(
                img_array, vit_feature_transformed, alpha=0.7, beta=0.3
            )
            img_combined = Image.fromarray(combined_feature)

            # Save processed features as PNG
            image_name = (
                f"{os.path.splitext(os.path.basename(image_file))[0]}_features.png"
            )
            output_path = os.path.join(output_folder, image_name)
            img_combined.save(output_path)
