# Multimodal Disease Prediction System
### Machine Learning | Computer Vision | NLP

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red.svg)
  ![NLP](https://img.shields.io/badge/NLP-TF--IDF-green.svg)
  ![CV](https://img.shields.io/badge/CV-ResNet50-yellow.svg)
  ![Accuracy](https://img.shields.io/badge/Text%20Accuracy-97%25-brightgreen.svg)
  ![Accuracy](https://img.shields.io/badge/Image%20Accuracy-85%25-brightgreen.svg)

  <img width="196" height="257" alt="Project Logo" src="https://github.com/user-attachments/assets/245ef033-d3f9-4925-81bf-c8169e071517" />

  <h3>A unified ML system for disease prediction using both textual symptoms and medical images</h3>
</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Text-Based Disease Prediction](#text-based-disease-prediction)
- [Image-Based Disease Prediction](#image-based-disease-prediction)
- [GUI Application](#gui-application)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Repository Files](#repository-files)
- [Installation](#installation)
- [Usage](#usage)
- [Future Scope](#future-scope)
- [Contributors](#contributors)

---

## Project Overview

This project is a Machine Learning-based system that predicts diseases using two different input modalities:

- Symptom-Based Prediction (NLP + Machine Learning)
- Image-Based Prediction (Computer Vision + Transfer Learning)

The system integrates both models into a unified pipeline with a custom-built GUI application for real-time predictions.

Key Features:
- Dual Input Modes: Accept both symptom text and medical images
- High Accuracy: 97% for text-based, 85% for image-based prediction
- User-Friendly GUI: Custom interface for easy interaction
- Real-Time Results: Instant predictions with confidence scores
- Transfer Learning: ResNet50 for image classification

What Users Can Do:
- Enter symptoms manually and get disease predictions
- Upload medical images for analysis
- View instant prediction results with confidence scores
- Seamless integration of both modalities

---

## Model Architecture

<div align="center">
  <img width="372" height="340" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/8491ac58-4005-4b69-803e-51acfb4e24cc" />
</div>

The system consists of two parallel pipelines that converge into a unified GUI:



---

## Text-Based Disease Prediction

**Pipeline:**
- Text preprocessing (cleaning, tokenization, stopword removal)
- TF-IDF vectorization for feature extraction
- Machine Learning models evaluated:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (Final Selected Model)

<div align="center">
  <img width="169" height="296" alt="Text Model Performance" src="https://github.com/user-attachments/assets/9e19efdd-7266-4147-b709-0ecea73d0da3" />
</div>

**Accuracy Achieved: 97%**

---

## Image-Based Disease Prediction

**Pipeline:**
- Transfer Learning using ResNet50 (pre-trained on ImageNet)
- Image preprocessing and augmentation techniques
- Fine-tuned classification layers
- Optimizers tested: ADAM, RMS-PROP
- Softmax output layer for multi-class classification

<div align="center">
  <img width="135" height="404" alt="Image Model Architecture" src="https://github.com/user-attachments/assets/881c7e5e-dee0-494d-916a-841aa1a120f4" />
</div>

**Accuracy Achieved: 85%**

---

## GUI Application

A custom Python-based GUI was developed to provide a seamless user experience:

Features:
- Text input field for symptom entry
- Image upload functionality for medical images
- Real-time prediction display
- Confidence score visualization
- Clean and intuitive interface

The GUI integrates both text and image models into a single unified interface, allowing users to choose their preferred input method.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.8+ |
| ML Framework (Text) | Scikit-learn |
| Deep Learning Framework | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Transfer Learning | ResNet50 |
| Data Processing | NumPy, Pandas |
| Text Vectorization | TF-IDF |
| GUI Framework | Tkinter / PyQt |
| Model Persistence | Joblib |

---

## Results

| Model Type | Accuracy |
|------------|----------|
| Text-Based Model (SVM) | 97% |
| Image-Based Model (ResNet50) | 85% |

The text-based model achieves exceptional accuracy due to clear symptom-disease correlations in the dataset. The image-based model shows good performance considering the complexity of medical image classification.

---

## Repository Files

This repository contains the following files:

| File Name | Description |
|-----------|-------------|
| `DiseasePredictionusingNLP.ipynb` | Jupyter notebook for text-based disease prediction model |
| `skinrash.ipynb` | Jupyter notebook for image-based skin disease classification |
| `predict_page.py` | GUI application script for real-time predictions |
| `patient_data.csv` | Dataset containing symptom-disease records |
| `tfidf_vectorizer_disease_nlp.joblib` | Saved TF-IDF vectorizer for text preprocessing |
| `label_encoder_disease_nlp.joblib` | Saved label encoder for disease classes |

**Note:** 
- The trained deep learning image model (.h5 file) is not uploaded because it exceeds GitHub's file size limit (100MB).
- Due to GitHub storage limits, trained model weights are excluded.
- Models can be regenerated by running the `skinrash.ipynb` file.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)


