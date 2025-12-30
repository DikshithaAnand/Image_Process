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
```text
User Uploads Image
↓
Streamlit Frontend (frontend/app.py)
↓
Image Preprocessing (Resize & Normalize)
↓
MobileNetV2 (Feature Extraction)
↓
Custom CNN Classification Layer
↓
Softmax Output (Probabilities)
↓
Prediction + Confidence / Unknown Handling
```

---

## 📁 Project Folder Structure

```text
Image_Tagging_Project/
├── dataset/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   └── dog/
│       ├── dog1.jpg
│       ├── dog2.jpg
│       └── ...
├── model/
│   └── image_tagger.h5
├── frontend/
│   └── app.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```


---

## ⚙️ How the System Works

1. Images are stored in folders where each folder name represents a class label.
2. Images are resized to **128 × 128** pixels and normalized.
3. MobileNetV2 extracts high-level image features.
4. A custom dense layer predicts the image class.
5. The Streamlit frontend allows users to upload images.
6. The model outputs prediction probabilities using Softmax.
7. If confidence is below a defined threshold, the image is classified as **Unknown**.

---

## 🧪 Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Learning Approach:** Transfer Learning
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Input Shape:** 128 × 128 × 3
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 3 (CPU-friendly training)

---

## 🖥️ Frontend Features

- Upload image via browser
- Display uploaded image
- Predict image category
- Show confidence percentage
- Handle unknown images gracefully
- Simple and user-friendly interface

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```bash
python train.py
```

## This will generate:

model/image_tagger.h5
### 3️⃣ Run the Frontend Application
```bash
cd frontend
streamlit run app.py
```

Open in browser:
```bash
http://localhost:8501
```

### ⚠️ Unknown Image Handling

If the prediction confidence is below 70%, the system displays:
```bash
⚠️ Unknown object (not cat or dog)
```

### 📊 Sample Output
```bash
Prediction: CAT
Confidence: 95.91%
```
## OR 

```bash
Unknown object (not cat or dog)
Confidence was only 48.20%
```
## 📈 Future Enhancements

- Add more image classes (human, car, animal types)
- Implement multi-label image tagging
- Deploy using Flask or FastAPI
- Add cloud deployment (AWS / GCP)
- Improve UI with custom CSS
- Integrate real-time camera input

---

## 🎓 Learning Outcomes

- Understanding of CNNs and transfer learning
- Practical experience with TensorFlow & Keras
- Frontend integration using Streamlit
- Real-world handling of unknown data
- GitHub project structuring and version control

---

## 📜 License

This project is developed for **educational purposes** and is free to use for learning and experimentation.

---

## 👤 Author

**Dikshitha A**  
Aspiring Software Engineer | AI & ML Enthusiast

