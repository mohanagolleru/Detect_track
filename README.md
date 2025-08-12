# Real-Time Object Detection and Tracking with YOLOv8

![Project Status](https://img.shields.io/badge/Status-Complete-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🎯 Project Overview

This project implements a custom object detection and tracking system for identifying specific objects—specifically "Nerf Guns" and "persons"—in video streams. The system is built on the powerful **YOLOv8** architecture, leveraging transfer learning to fine-tune a pre-trained model on a custom dataset.

The core goal of this project was to develop a robust system capable of handling real-world challenges such as motion blur, varying lighting conditions, and partial occlusion. The final model demonstrates high detection accuracy and is deployed to perform real-time inference on new video data.

---

### Key Components and Methodology

This project utilizes a clear, step-by-step methodology to achieve high-performance object detection.

#### 1. Environment Setup and Library Installation

The project begins by setting up the development environment and installing necessary libraries.

* `import os`: Imports the operating system module for file system interaction.
* `HOME = os.getcwd()`: Sets the `HOME` variable to the current working directory, which is typically `/content` in a Google Colab environment.
* `!pip install ultralytics==8.0.20`: Installs the **Ultralytics** library, which provides the **YOLOv8 model**. The specific version is pinned to ensure reproducibility.
* `ultralytics.checks()`: Verifies that all dependencies, including Python, PyTorch, and CUDA, are correctly installed and configured.

#### 2. Dataset Preparation with Roboflow

A custom dataset is prepared and downloaded using the **Roboflow** platform.

* `!pip install roboflow`: Installs the Roboflow Python library.
* `rf = Roboflow(api_key="...")`: Initializes a Roboflow object using an API key to access the private dataset.
* `project = rf.workspace("su-ujhvp").project("3-wbfyt")`: Connects to a specific project within the Roboflow workspace.
* `dataset = project.version(3).download("yolov8")`: Downloads the third version of the project's dataset in **YOLOv8 format**, which includes images annotated with two classes: "Nerf Gun" and "person."

#### 3. Transfer Learning and Model Fine-Tuning

This is the core of the object detection process, where a pre-trained model is adapted for a new task.

* `!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=120 imgsz=640 plots=True`: This command starts the training process.
    * **`yolo`**: Invokes the Ultralytics YOLO command-line interface.
    * **`task=detect`**: Specifies the task as object detection.
    * **`mode=train`**: Puts the YOLO model in training mode.
    * **`model=yolov8s.pt`**: The base model, a small, pre-trained YOLOv8 model, is used as the starting point for **transfer learning**. This enables the model to leverage features already learned from a large dataset, making the fine-tuning process faster and more efficient.
    * **`data={dataset.location}/data.yaml`**: Points to the configuration file defining the dataset's location and class names.
    * **`epochs=120`**: Sets the number of training cycles over the entire dataset to 120.
    * **`imgsz=640`**: Defines the input image size as 640x640 pixels.
    * **`plots=True`**: Generates and saves various performance plots and metrics.

#### 4. Model Validation and Performance Metrics

After training, the model's performance is evaluated using a separate validation set.

* `!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml`: This command runs the validation mode.
    * **`mode=val`**: Puts the YOLO model in validation mode to evaluate its performance on unseen data.
    * **`model=.../weights/best.pt`**: Specifies the path to the best-performing model checkpoint saved during training.
* The output shows a **mean Average Precision (mAP) of 86.1%** at an Intersection over Union (IoU) of 50% (`mAP@50`), and a **mAP of 57.8%** across IoU thresholds from 50% to 95% (`mAP@50-95`). This is the key metric demonstrating the model's high accuracy.
* 
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




