# 🩸 RedShift: Malaria Infected RBC Detector

RedShift is a deep learning-based web application built with **Streamlit** that detects malaria infection in red blood cells (RBCs) from microscopic images. It uses a Convolutional Neural Network (CNN) to classify cells as **Parasitized** or **Uninfected**, aiding medical diagnostics.

---

## 🚀 Features

- 📷 Upload and analyze cell images in real-time
- 🧠 CNN-based deep learning model
- 🧼 Image preprocessing for improved accuracy
- 🎨 Clean, interactive Streamlit interface
- 💾 Model stored and loaded via `.keras` format

---

## 🧬 Model Summary

The CNN model includes:
- 3 Convolutional layers with ReLU activations
- MaxPooling layers after each convolution
- Flattening followed by Fully Connected Dense layers
- Sigmoid output layer for binary classification

---

## 📂 Dataset

- **Source**: [NIH Malaria Dataset on Kaggle]
- **Classes**:
  - `Parasitized`
  - `Uninfected`

---

## 🛠 Technology Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **PIL (Pillow)**
- **Streamlit**
