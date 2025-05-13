# **Comprehensive Documentation for Rice Disease Detection Project**

#### Project Overview

The **Rice Disease Detection** project aims to develop an automated computer-aided diagnosis tool to classify rice plant diseases from images. The application utilizes a deep learning model to detect three types of diseases: **Leaf Smut, Brown Spot, and Bacterial Leaf Blight.**

#### Table of Contents

1. [Data Acquisition](#1-data-acquisition)
2. [Data Pre-Processing](#2-data-pre-processing)
3. [Model Development](#3-model-development)
4. [Dataset Splitting](#4-dataset-splitting)
5. [Model Training &amp; Validation](#5-model-training--validation)
6. [Model Evaluation](#6-model-evaluation)
7. [Real-Time Detection Pipeline](#7-real-time-detection-pipeline)
8. [Web Application Development](#8-web-application-development)
9. [Testing &amp; Deployment](#9-testing--deployment)
10. [Run Instructions](#10-run-instructions)
11. [Summary](#11-summary)

---

#### 1. Data Acquisition

**Task 1.1: Identify and Select Data Sources**

- Search for free repositories that provide labeled images of rice plant diseases.

**Task 1.2: Download and Organize Images**

- Images are organized into three folders based on disease type:
  - Bacterial Leaf Blight
  - Leaf Smut
  - Brown Spot

**File:** dataset/

---

#### 2. Data Pre-Processing

**Task 2.1: Image Resizing, Normalization, and Formatting**

- Resize images to a uniform size (128x128 pixels) and normalize pixel values to the range [0, 1].

**Task 2.2: Data Augmentation**

- Apply techniques like flipping and rotation to enhance dataset diversity.

**Task 2.3: Save Preprocessed Data**

- Save the processed images into structured training, validation, and test folders.

**File:** data_preprocessing.py

---

#### 3. Model Development

**Task 3.1: Research CNN Architectures**

- Evaluate various architectures like VGG16, ResNet, and MobileNet.

**Task 3.2: Select Model Architecture**

- Chose MobileNetV2 for its efficiency and accuracy in image classification.

**File:** train_model.py

---

#### 4. Dataset Splitting

**Task 4.1: Split the Dataset**

- Split the dataset into 70% training, 15% validation, and 15% testing.

**File:** Handled in data_preprocessing.py.

---

#### 5. Model Training & Validation

**Task 5.1: Train the CNN Model**

- Train the selected model using the training dataset.

**Task 5.2: Validate Model Accuracy**

- Use the validation set to check the model's performance.

**Task 5.3: Hyperparameter Tuning**

- Adjust learning rates, epochs, and batch sizes to optimize performance.

**File:** train_model.py

---

#### 6. Model Evaluation

**Task 6.1: Evaluate the Final Model**

- Use the test dataset to evaluate model performance.

**Task 6.2: Performance Metrics**

- Calculate accuracy, precision, recall, F1-score, and confusion matrix.

**File:** evaluate_model.py

---

#### 7. Real-Time Detection Pipeline

**Task 7.1: Integrate OpenCV**

- Enable image upload and preprocessing using OpenCV.

**Task 7.2: Apply the Trained Model**

- Classify uploaded images using the trained model.

**File:** real_time_detection.py

---

#### 8. Web Application Development

**Task 8.1: Design Frontend Interface**

- Use Flask with HTML/CSS for the web interface.

**Task 8.2: Implement Functionality**

- Allow users to upload images and display results.

**Task 8.3: Visual Feedback**

- Show original images alongside classification results.

**File:** index.html

---

#### 9. Testing & Deployment

**Task 9.1: Conduct Testing**

- Perform unit and integration testing on the full pipeline.

**Task 9.2: Deploy the Web App**

- Deploy locally or on cloud platforms like Heroku or AWS.

**File:** app.py

---

#### 10. Run Instructions

1. Open the existing directory in your terminal which we worked on.
2. Install dependencies:
   pip install -r requirements.txt
3. Preprocess dataset:
   python data_preprocessing.py
4. Train the model:
   cd model
   python train_model.py
5. Evaluate the model:
   python evaluate_model.py
6. Run real time prediction:
   python real_time_prediction.py
7. Run the web app:
   python app.py

After all, as a final step ,upload an image to see the prediction.

---

> #### 11. Summary

This documentation provides a comprehensive overview of the Rice Disease Detection project, detailing each step taken to create the application. Each file and its purpose are explained, making it beginner-friendly and informative.

---
