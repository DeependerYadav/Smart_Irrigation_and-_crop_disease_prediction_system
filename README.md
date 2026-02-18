# 🌱 Integrated Smart Irrigation & Crop Disease Prediction System

## 🚀 AI-Powered Precision Agriculture Platform

---

## 📌 Overview

The **Integrated Smart Irrigation & Crop Disease Prediction System** is an AI-driven agricultural decision-support platform designed to optimize irrigation practices and enable early crop disease detection. By leveraging deep learning and environmental data analysis, the system provides intelligent, real-time recommendations to improve water efficiency and crop health.

This project aims to promote sustainable farming through data-driven decision-making and smart automation.

---

## 🎯 Problem Statement

Agriculture faces two major challenges:

- **Inefficient Irrigation** – Traditional irrigation methods rely on fixed schedules rather than real-time environmental conditions, leading to water wastage.
- **Late Disease Detection** – Crop diseases are often identified only after visible damage occurs, resulting in reduced yield and economic losses.

There is a need for a scalable, AI-powered solution that can:

- Optimize water usage
- Detect crop diseases early
- Improve overall agricultural productivity

---

## 💡 Proposed Solution

Our system integrates:

- Environmental parameters (soil moisture, temperature, humidity, rain forecast)
- Crop type input
- Leaf image analysis
- Deep learning models
- Interactive web dashboard

### Key Components:

### 🚿 Irrigation Prediction Model
- Multi-layer Neural Network (Regression)
- Predicts optimal water requirement (liters)
- Uses feature scaling and nonlinear learning
- Reduces over-irrigation and water wastage

### 🌿 Crop Disease Detection Model
- Transfer Learning using MobileNetV2
- CNN-based image classification
- Predicts disease type with confidence score
- Enables early intervention

### 📊 Web Dashboard
- Built using Streamlit
- Real-time prediction interface
- Displays irrigation recommendation and disease analysis
- Calculates water efficiency score

---

## 🏗 System Architecture
Input Data
↓
Environmental Parameters + Leaf Image
↓
Machine Learning Models
↓
Prediction Output
↓
Streamlit Dashboard Interface


## Files

- `app.py` - Streamlit application
- `irrigation_training.py` - train irrigation model
- `irrigation_predict.py` - irrigation prediction helper
- `disease_training.py` - train cotton disease model
- `disease_predict.py` - cotton disease prediction helper
- `irrigation_model.h5` - irrigation model artifact
- `scaler.pkl` - irrigation feature scaler
- `disease_model.h5` - cotton disease model artifact
- `labels.pkl` - cotton class labels

## Cotton Dataset Path

Default path used by training script:

`C:\Users\HP\Downloads\data for cotton\Cotton Disease`

Expected structure:

```text
Cotton Disease/
  train/
    diseased cotton leaf/
    diseased cotton plant/
    fresh cotton leaf/
    fresh cotton plant/
  val/
    ...
  test/
    ...
```

## Installation

```bash
python -m venv .venv
# cmd
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Train Models

Irrigation model:

```bash
python irrigation_training.py
```

Cotton disease model:

```bash
python disease_training.py --epochs 8 --fine-tune-epochs 2
```

##Results

High regression accuracy for irrigation prediction

Accurate disease classification using transfer learning

Reduced over-irrigation through AI-based recommendation

Improved early-stage disease identification


##Our system promotes:

 Water conservation

 Increased crop yield

 Reduced economic loss

 Sustainable agriculture


## Run App

```bash
https://deependeryadav-smart-irrigation-and--crop-disease-pr-app-9jayzf.streamlit.app/
```
