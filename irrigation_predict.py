"""Prediction helper for irrigation neural-network model."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import tensorflow as tf


MODEL_PATH = Path(__file__).with_name("irrigation_model.h5")
SCALER_PATH = Path(__file__).with_name("scaler.pkl")
FEATURE_COLUMNS = ["soil_moisture", "temperature", "humidity", "rain_forecast", "crop_type"]

_MODEL = None
_SCALER = None


def _load_artifacts():
    """Load and cache model + scaler."""
    global _MODEL, _SCALER
    if _MODEL is not None and _SCALER is not None:
        return _MODEL, _SCALER

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Irrigation model not found at {MODEL_PATH}. Run irrigation_training.py first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found at {SCALER_PATH}. Run irrigation_training.py first."
        )

    _MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    _SCALER = joblib.load(SCALER_PATH)
    return _MODEL, _SCALER


def _prepare_input(input_dict: dict[str, Any]) -> pd.DataFrame:
    """Validate and format input payload."""
    required = ["soil_moisture", "temperature", "humidity", "rain_forecast", "crop_type"]
    missing = [k for k in required if k not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    try:
        soil_moisture = float(input_dict["soil_moisture"])
        temperature = float(input_dict["temperature"])
        humidity = float(input_dict["humidity"])
        rain_forecast = int(input_dict["rain_forecast"])
        crop_type = float(input_dict["crop_type"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid input values: {exc}") from exc

    if rain_forecast not in (0, 1):
        raise ValueError("rain_forecast must be 0 or 1")

    return pd.DataFrame(
        [[soil_moisture, temperature, humidity, rain_forecast, crop_type]],
        columns=FEATURE_COLUMNS,
        dtype=float,
    )


def predict_irrigation(input_dict: dict[str, Any]) -> float:
    """Return predicted water requirement in liters."""
    model, scaler = _load_artifacts()
    features = _prepare_input(input_dict)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled, verbose=0)
    return float(prediction[0][0])


if __name__ == "__main__":
    sample_input = {
        "soil_moisture": 40,
        "temperature": 32,
        "humidity": 60,
        "rain_forecast": 0,
        "crop_type": 2,
    }
    print(f"Predicted water required: {predict_irrigation(sample_input):.2f} liters")
