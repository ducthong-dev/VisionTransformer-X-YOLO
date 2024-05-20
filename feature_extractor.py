import torch
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import time
import numpy as np

# Load the ViT feature extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load image
image = Image.open("dataset/train/Apple___Apple_scab/image (2).JPG")
# .convert("RGB")
print(image)
image.show()
exit()

# Extract features from the image
start_time = time.time()
inputs = feature_extractor(images=image, return_tensors="pt")
end_time = time.time()
processing_time = end_time - start_time
print("Processing time:", processing_time, "seconds")
# Retrieve the extracted features
features = inputs.pixel_values

# Print the shape of the extracted features
print(features.shape)  # Output: torch.Size([1, 196, 768])


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
image_proccessed = transform(features[0])
image_proccessed.show()

# image_proccessed.save("image.png")
