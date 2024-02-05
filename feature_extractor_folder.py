import torch
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import time
import numpy as np
import os

# Load the ViT image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


def extract_features(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Retrieve the extracted features
    features = inputs.pixel_values

    return features


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
    return transform(features[0])


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


ROOT_FOLDER = "dataset/Plant_leave_diseases_dataset_without_augmentation"
all_classes_names = read_folder_names(ROOT_FOLDER)

for class_name in all_classes_names:
    print(class_name)
    image_folder = f"{ROOT_FOLDER}/{class_name}"
    output_folder = f"/Users/ducthong/Desktop/AI/Computer Vision/Vision Transformer/save/{class_name}"
    remove_files_in_folder(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        features = extract_features(image_path)
        image = transform_image(features=features)

        # Save processed features as PNG
        image_name = f"{os.path.splitext(os.path.basename(image_file))[0]}_features.png"
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)
