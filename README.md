# Brain Tumor Detection Project

This repository contains the code and resources for a Brain Tumor Detection project using Convolutional Neural Networks (CNN) with PyTorch. The project involves analyzing CT images of the brain to differentiate between healthy and tumor-affected images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Training and Evaluation](#training-and-evaluation)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

The Brain Tumor Detection project aims to accurately classify brain CT images as either healthy or tumor-affected. This project leverages deep learning techniques using PyTorch to build and train a CNN model. The trained model is then deployed in a Flask-based web application to provide real-time classification of brain CT images.

## Dataset

The dataset used for this project consists of CT images of the brain, sourced from Kaggle. It includes images of both healthy and tumor-affected brains.

## Project Structure
brain-tumor-detection/
├── __pycache__/
├── model/ 
│ └── model.pth
├── static/
│ ├── css/
│ └── uploads/
├──templates/
│ ├── detect.html
│ └── index.html
├── README.md
├── tumor.py
├── app.py
├──Brain Tumor Detection.pptx
└──Tumor Detector.ipynb


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. Install the required packages:
```
    requirements: 
     python
     pyTorch(visual) (checkout from their website for others libraries)
     numpy
     pandas
     sklearn
     matplotlib
     seaborn
     flask
sh
    pip install -r requirements.txt
```

## Usage

1. Preprocess the data:
    ```sh
    cd notebooks
    jupyter Tumor Detector.ipynb.ipynb
    ```

2. Train the model:
    ```sh
    jupyter Tumor Detector.ipynb.ipynb
    ```

3. Run the web application:
    flask run 

## Model Development

The CNN model is built using PyTorch. Key components include:
- Data preprocessing using NumPy and Pandas for resizing and format standardization.
- Custom dataset and dataloader classes to manage image data.
- A CNN model for classifying brain images.

## Training and Evaluation

The model is trained over approximately 600 epochs. The training process involves:
- Splitting the dataset into training and validation sets.
- Using data augmentation techniques to improve model generalization.
- Monitoring the model's performance and checking for overfitting.

## Web Application

The trained model is deployed in a Flask-based web application. The web app allows users to upload brain CT images and receive real-time classification results, indicating whether the image depicts a healthy brain or one with a tumor.

## Results

The model's performance is evaluated based on metrics such as accuracy, precision, recall, and F1-score. Graphical representations of the results are provided using Matplotlib and Seaborn.

## Contributing
Contributions are welcome.


## Preview
![tumor Web](https://github.com/hamzaanwar12/Tumor-Detector-Using-Pytorch/assets/147822744/64778340-286a-4d63-a22c-586b4934a550)

