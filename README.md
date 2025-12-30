# 🐱🐶 Image Tagging System using TensorFlow & Streamlit

An end-to-end **Image Tagging (Image Classification) system** built using **TensorFlow (CNN + Transfer Learning)** and deployed with a **Streamlit frontend**.  
The application allows users to upload an image and receive a predicted label along with a confidence score.

---

## 📌 Project Overview

This project implements an **image tagging system** that classifies uploaded images into predefined categories such as **Cat** and **Dog**.  
A **pre-trained MobileNetV2** model is used for feature extraction, making the system **CPU-efficient** and suitable for execution on local machines.

To improve real-world reliability, the application includes **confidence-based unknown detection**, preventing incorrect predictions for unrelated images (for example, human faces).

---

## 🎯 Objectives

- Develop an image classification model using Convolutional Neural Networks (CNN)
- Apply **transfer learning** to reduce training time and CPU usage
- Preprocess and normalize image data
- Deploy the trained model using a frontend web interface
- Display prediction results with confidence scores
- Handle unknown or unrelated images intelligently

---

## 🧠 Technologies Used

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow & Keras  
- **Pre-trained Model:** MobileNetV2  
- **Frontend Framework:** Streamlit  
- **Image Processing:** Pillow (PIL), NumPy  
- **Version Control:** Git & GitHub  

---

## 🏗️ System Architecture

- User Uploads Image
↓
- Streamlit Frontend (frontend/app.py)
↓
- Image Preprocessing (Resize & Normalize)
↓
- MobileNetV2 (Feature Extraction)
↓
- Custom CNN Classification Layer
↓
- Softmax Output (Probabilities)
↓
- Prediction + Confidence / Unknown Handling
