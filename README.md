## Plant Disease Classification with YOLOv8n-cls (ViT Features)

This repository contains the code and configuration files for training a YOLOv8n-cls model with Vision Transformer (ViT) features for plant disease classification. The model is trained from scratch on Google Colab Pro using a custom dataset of pre-extracted features.

### Project Goals

* Leverage YOLOv8n-cls architecture, optimized for classification tasks, for plant disease identification.
* Utilize pre-extracted ViT features from plant images to improve model performance.
* Train the model from scratch to learn optimal parameters for the specific plant disease classification task.

### Requirements

* Python (tested with python 3.11.5)
* PyTorch (tested with 2.2.0)
* Ultralytics YOLOv8 (from source or [https://github.com/ultralytics](https://github.com/ultralytics))
* Additional libraries specified in `requirements.txt`

### Getting Started

1. Clone this repository:

```bash
git clone https://github.com/ducthong-dev/VisionTransformer-X-YOLO.git
```

2. Install dependencies:

```bash
cd VisionTransformer-X-YOLO
pip install -r requirements.txt
```

3. Download the pre-extracted features dataset (not included in this repository due to size considerations).

4. Update the data path in `yolov8n_cls.yaml` to point to your downloaded dataset directory.

**Note:** Training can be computationally expensive. Consider using Google Colab Pro for faster training times.

### Training Configuration

The training configuration is defined in `yolov8n_cls.yaml`. Key parameters include:

* Model: YOLOv8n-cls (custom variant for classification)
* Number of Classes (nc): Overridden to match the number of plant disease classes in your dataset
* Batch Size: 128
* Image Size: 224x224 (adjusted to the size of your pre-extracted features)
* Optimizer: Adaptive (potentially Adam or SGD with momentum)
* Learning Rate: 0.01 (initial and final)
* Epochs: 50
* Data Augmentation: Mosaic (coefficient 1.0), Random Augmentation (`randaugment`), Erasing (coefficient 0.4)

### Training Results

The training log (`runs/train/train.log`) provides detailed information about the training process, including:

* Training and validation losses
* Learning rate curve
* Training time (approximately 4 hours on Google Colab Pro for 50 epochs)

**Note:** You may need to adjust hyperparameters and training configuration based on your specific dataset and hardware resources.

### Additional Notes

* This project demonstrates training YOLOv8n-cls with ViT features for plant disease classification from scratch. 
* You can further explore fine-tuning a pre-trained YOLOv8 model with your dataset for potentially faster convergence.
* Experiment with different data augmentation techniques and hyperparameter tuning to improve model performance.

### License

This project is licensed under the MIT License (see LICENSE file).
