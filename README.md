# Real-Time Object Detection and Tracking with YOLOv8

![Project Status](https://img.shields.io/badge/Status-Complete-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🎯 Project Overview

This project implements a custom object detection and tracking system for identifying specific objects—specifically "Nerf Guns" and "persons"—in video streams. The system is built on the powerful **YOLOv8** architecture, leveraging transfer learning to fine-tune a pre-trained model on a custom dataset.

The core goal of this project was to develop a robust system capable of handling real-world challenges such as motion blur, varying lighting conditions, and partial occlusion. The final model demonstrates high detection accuracy and is deployed to perform real-time inference on new video data.

---

## ✨ Key Features

* **YOLOv8 Architecture**: Utilizes the state-of-the-art Ultralytics YOLOv8 model, a highly efficient and accurate object detection framework.
* **Transfer Learning**: Fine-tuned a pre-trained `yolov8s.pt` model on a custom dataset to accelerate training and optimize performance for specific object classes.
* **Custom Data Integration**: The model was trained on a specialized dataset containing images of "Nerf Guns" and "persons," sourced from Roboflow.
* **Hyperparameter Optimization**: Key hyperparameters, including IoU thresholds and confidence scores, were carefully tuned during the training process to achieve a high detection rate and minimize false positives.
* **Data Augmentation**: A range of data augmentation techniques (e.g., blurring, grayscale conversion, and contrast adjustments via CLAHE) were applied to improve the model's generalization capabilities and robustness to diverse visual conditions.
* **High Performance**: The final model achieved a mean Average Precision (**mAP@50-95 of 57.8%**) on the validation set, with an impressive **mAP@50 of 91.8%** for the "Nerf Gun" class.
* **Real-Time Inference**: The trained model can perform rapid object detection on new video streams, demonstrating its utility for real-world applications.

---

## ⚙️ Technical Workflow

The project's workflow is organized into four main stages:

1.  **Environment Setup**: Dependencies such as `ultralytics` and `roboflow` are installed, and the environment is verified to ensure compatibility with a CUDA-enabled GPU.
2.  **Dataset Preparation**: The custom dataset is programmatically downloaded from Roboflow and automatically configured for YOLOv8 training.
3.  **Model Fine-Tuning**: A `yolov8s.pt` model is trained for 120 epochs. The training process generates performance plots and saves the best model weights.
4.  **Validation & Inference**: The best-performing model is validated to confirm its accuracy and then used to detect objects in a new video file. The processed video, with bounding boxes and class labels, is saved to Google Drive.

---

## 🚀 Getting Started

To replicate this project, you will need a Google Colab environment with GPU access.

### Prerequisites

* A Google account with access to Google Colab.
* A Roboflow API key to download the custom dataset.

### Installation and Setup

1.  Open the provided Jupyter Notebook in Google Colab.
2.  Run the first code block to install `ultralytics` and other required libraries.
3.  Replace the placeholder API key with your own Roboflow API key to access the dataset.
4.  Mount your Google Drive to save the processed video.
5.  Execute the remaining cells sequentially to perform training, validation, and inference.

---

## 📊 Results and Performance

The training process generated a series of performance plots and metrics. Key metrics from the validation stage are as follows:

| Class | mAP@50 | mAP@50-95 |
|---|---|---|
| **All** | 0.906 | 0.578 |
| **Nerf Gun** | 0.918 | 0.550 |
| **person** | 0.894 | 0.606 |

These results confirm that the fine-tuned model performs exceptionally well in detecting the custom object classes.




