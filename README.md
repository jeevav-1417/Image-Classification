# Image-Classification
This project is an Image Classification System using MobileNetV2 and a custom CIFAR-10 CNN. It includes a Streamlit app where users upload images, select models, and view predictions with confidence scores. MobileNetV2 handles diverse real-world images, while CIFAR-10 CNN classifies into 10 categories. Ideal for exploring AI workflows!

# 🖼️ Image Classification System

This repository contains an **Image Classification System** developed using deep learning techniques. It supports two models:

1. **MobileNetV2**: A pre-trained model on ImageNet for general image classification.
2. **CIFAR-10 CNN**: A custom-trained Convolutional Neural Network for CIFAR-10 dataset classification.

## 🚀 Features

- **CIFAR-10 Training**:
  - Implements both ANN and CNN models.
  - Visualizes training and validation loss/accuracy.
  - Saves the trained CNN model for reuse.

- **MobileNetV2 ImageNet Classification**:
  - Leverages a pre-trained MobileNetV2 model.
  - Processes and classifies user-uploaded images into ImageNet categories.

- **Streamlit Web Application**:
  - User-friendly interface for image uploads and model selection.
  - Interactive classification results and download options for predictions.


## 🧠 Models Used

### 1. MobileNetV2 (ImageNet)
- **Purpose**: General image classification.
- **Preprocessing**: Resize to 224x224, MobileNetV2 normalization.
- **Predictions**: Returns top-3 predicted classes with confidence scores.

### 2. CIFAR-10 CNN
- **Purpose**: Classify images into 10 categories: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.
- **Architecture**:
  - Convolutional layers with ReLU activation and MaxPooling.
  - Fully connected layers with dropout for regularization.
  - Trained with 10 epochs using Adam optimizer.

## 🖥️ How to Run

### Prerequisites
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt

##📊 Visualizations
Training and validation loss/accuracy plots for CIFAR-10 models.
Interactive results with confidence scores for both models.
📈 Training Results
CIFAR-10 CNN achieved XX% validation accuracy after 10 epochs.
✨ Example Usage
CIFAR-10 Classification
Upload a 32x32 image to classify into one of the 10 categories.
View the predicted class with confidence.
MobileNetV2 Classification
Upload a high-resolution image.
Get the top-3 ImageNet class predictions with confidence.
🤝 Contributing
Feel free to raise issues or submit pull requests to improve this project.

