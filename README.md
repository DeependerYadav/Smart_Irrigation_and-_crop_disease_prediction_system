# Smart Irrigation & Cotton Disease Prediction System

This project has two AI modules:

1. Irrigation prediction (TensorFlow neural network regression)
2. Cotton disease detection (EfficientNetB0 with MobileNetV2 fallback)

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
