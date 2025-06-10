# Recycling-AI
This repository hosts an AI-driven waste detection and classification toolkit. It provides easy-to-use scripts for training a model on your own waste image dataset and running inference on photos to automatically sort items into categories like plastic, paper, glass, metal, and other

# AI-Powered Waste Detection and Classification

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation & Setup](#installation--setup)  
   - [Windows & WSL2 Setup](#windows--wsl2-setup)  
   - [CUDA & cuDNN](#cuda--cudnn)  
   - [CPU-only Execution](#cpu-only-execution)  
5. [Directory Structure](#directory-structure)  
6. [Usage](#usage)  
   - [Training](#training)  
   - [Real-Time Detection (CLI & Web)](#real-time-detection-cli--web)  
7. [Dataset](#dataset)  
8. [Contact](#contact)

---

## Project Overview
This project develops an AI system to automate waste detection and classification. It processes **static images** to recognize materials like plastic, paper, glass, and metal.

## Features
- **Static image classification**  
- **Multi-class detection** (Plastic, Paper, Glass, Metal, Organic)  
- **GPU-accelerated training and inference**  

## Prerequisites
- **OS**: Windows 10/11  
- **WSL2**: Ubuntu 22.04  
- **GPU**:  For best performance please use RTX  40x , 30x  series (or better)  
- **CUDA**: 12.2  
- **cuDNN**: 8.9  
- **Python**: 3.10  

## Installation & Setup

### Windows & WSL2 Setup
1. Enable WSL2:
   ```powershell
   wsl --install
   ```
2. Confirm installation:
   ```bash
   wsl -l -v
   ```
3. Install **Ubuntu 22.04** from Microsoft Store.  
4. Install **VS Code Remote - WSL** extension.

### CUDA & cuDNN
1. Install **NVIDIA GPU Driver** (Game Ready / Studio Driver) in Windows.  
2. Inside Ubuntu:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y cuda-toolkit-12-2
   ```
3. Download cuDNN 8.9, extract, and copy files:
   ```bash
   sudo cp cudnn-8.9.*/include/cudnn*.h /usr/local/cuda-12.2/include/
   sudo cp cudnn-8.9.*/lib/libcudnn*.so* /usr/local/cuda-12.2/lib64/
   ```
4. Set environment variables:
   ```bash
   echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

### CPU-only Execution
If you want to force CPU usage instead of GPU:

Training:
```bash
python model.py --train --data_dir waste/ --epochs 50 --device cpu
```

Inference (single image):
```bash
python cam.py --image path/to/image.jpg --device cpu
```

## Directory Structure
```
RecyclingProject/
├──Web                         # Web components and interface
<├──testData
├──dataset_resizer             # For resize dataset iamges to valid resolation
├── waste/                     # Raw and labeled datasets
│   ├── Paper/
│   ├── Plastic/
│   └── (Other categories)
├── README.md                   # This README file
├── model.py                    # Model training and saving script
├── EfficentNet80_cam.py        # Real-time detection using webcam
├── .gitignore
├── EfficentNet80_image.py      # Using single image detection    
```

## Usage

### Training
Train a new model from scratch:
```bash
python model.py --train --data_dir waste/ --epochs 50
```

### Real-Time Detection (CLI & Web)
Use webcam to detect materials in real-time:
```bash
EfficentNet80_cam.py or EfficentNet80_image.py
```
## 🌐 Web Interface

A lightweight web interface built with Flask is included to allow real-time interaction with the AI waste classification system. Users can upload images for classification or contribute labeled data to improve the dataset.

### 🔗 Available Routes

- **`/`** – Home page with project overview and navigation buttons  
- **`/predict`** – Upload a waste image and receive predicted label + confidence score  
- **`/contribute`** – Manually upload labeled images to grow the local dataset  
- **`/about`** – Project description, goals, and technologies used  

### 🛠️ Tech Stack

| Component           | Technology                          |
|---------------------|--------------------------------------|
| Web Framework       | Flask (Python 3.11)                  |
| Frontend            | HTML5, Bootstrap 5.3                 |
| AI Model Integration| TensorFlow 2.19.0 + EfficientNetB0  |
| Image Processing    | OpenCV (cv2), NumPy                  |
| Data Logging        | CSV file (`metadata.csv`)            |

### 🚀 Run Locally

```bash
# Make sure virtual environment and dependencies are installed
python app.py


## Dataset
- Dataset sourced from Kaggle, Google Open Images, etc.
- Images are categorized into: **Plastic**, **Paper**, **Glass**, **Metal**, **Others**.
- Dataset folder is `waste/`.  


## Contact
For any issues or questions, please reach out:
- celilbus@outlook.com
- furkansaribal@gmail.com
- gidisburak@gmail.com


