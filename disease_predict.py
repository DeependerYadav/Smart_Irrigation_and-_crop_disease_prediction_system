"""Prediction helper for cotton disease model."""

from pathlib import Path
from typing import Union
import io
import pickle

import numpy as np
from PIL import Image
import tensorflow as tf


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "disease_model.h5"
LABELS_PATH = BASE_DIR / "labels.pkl"
IMAGE_SIZE = (224, 224)

_MODEL = None
_LABELS = None


FRIENDLY_LABELS = {
    "diseased cotton leaf": "Diseased Cotton Leaf",
    "diseased cotton plant": "Diseased Cotton Plant",
    "fresh cotton leaf": "No Disease (Healthy Leaf)",
    "fresh cotton plant": "No Disease (Healthy Plant)",
}


def get_cotton_disease_reference() -> list[str]:
    """Names shown in UI for cotton disease outcomes."""
    return [
        "No Disease (Healthy Leaf)",
        "No Disease (Healthy Plant)",
        "Diseased Cotton Leaf",
        "Diseased Cotton Plant",
    ]


def _load_artifacts():
    """Load and cache model + class labels."""
    global _MODEL, _LABELS
    if _MODEL is not None and _LABELS is not None:
        return _MODEL, _LABELS

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Disease model not found at {MODEL_PATH}. Run disease_training.py first."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found at {LABELS_PATH}. Run disease_training.py first."
        )

    custom_objects = {
        "RandomZoom": tf.keras.layers.RandomZoom,
        "RandomHeight": tf.keras.layers.RandomHeight,
        "RandomWidth": tf.keras.layers.RandomWidth,
        "RandomFlip": tf.keras.layers.RandomFlip,
    }
    _MODEL = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects=custom_objects,
        compile=False,
    )
    with open(LABELS_PATH, "rb") as file_obj:
        _LABELS = pickle.load(file_obj)
    return _MODEL, _LABELS


def _load_image(image_input: Union[str, bytes, bytearray, Image.Image]) -> Image.Image:
    """Normalize accepted image inputs into one RGB PIL image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    return Image.open(str(image_input)).convert("RGB")


def _preprocess_loaded_image(image: Image.Image) -> np.ndarray:
    """Convert already-loaded image to model-ready tensor."""
    resized = image.resize(IMAGE_SIZE)
    arr = np.asarray(resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    # Preprocessing is handled inside the trained model graph.
    return arr


def preprocess_image(image_input: Union[str, bytes, bytearray, Image.Image]) -> np.ndarray:
    """Convert image input to model-ready tensor of shape (1, 224, 224, 3)."""
    return _preprocess_loaded_image(_load_image(image_input))


def format_prediction_label(raw_label: str) -> str:
    """Return user-friendly label and enforce healthy naming."""
    clean = raw_label.strip().lower()
    if clean in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[clean]

    pretty = raw_label.replace("_", " ").title()
    if "fresh" in clean or "healthy" in clean:
        if "leaf" in clean:
            return "No Disease (Healthy Leaf)"
        return "No Disease (Healthy Plant)"
    return pretty


def _predict_from_loaded_image(image: Image.Image) -> tuple[str, float]:
    """Run classifier on a loaded image and return label + confidence."""
    model, labels = _load_artifacts()
    x = _preprocess_loaded_image(image)
    probs = model.predict(x, verbose=0)[0]
    predicted_index = int(np.argmax(probs))
    raw_label = str(labels[predicted_index])
    confidence = float(probs[predicted_index])
    return format_prediction_label(raw_label), confidence


def estimate_diseased_area_from_image(image: Image.Image) -> float:
    """Estimate percentage of affected plant area using a color heuristic."""
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(resized).astype("float32")
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Leaf-like vegetation pixels.
    exg = (2.0 * g) - r - b
    vegetation_mask = (g > 38.0) & (g > (r * 0.78)) & (g > (b * 0.78)) & (exg > 6.0)

    # Candidate lesion colors: yellow/brown/reddish spots common on infected leaves.
    yellow_brown = (r > 85.0) & (g > 65.0) & (b < 135.0) & (r > (b * 1.10))
    red_brown = (r > 70.0) & (r > (g * 1.08)) & (r > (b * 1.08))
    dark_lesion = ((r + g + b) < 195.0) & (r > 20.0) & (g > 20.0)
    lesion_candidates = yellow_brown | red_brown | dark_lesion

    # Keep lesions that are on/near vegetation to reduce skin/background noise.
    near_vegetation = vegetation_mask.copy()
    for shift_y in (-1, 0, 1):
        for shift_x in (-1, 0, 1):
            if shift_x == 0 and shift_y == 0:
                continue
            shifted = np.roll(np.roll(vegetation_mask, shift_y, axis=0), shift_x, axis=1)
            near_vegetation |= shifted

    lesion_mask = lesion_candidates & near_vegetation
    plant_mask = vegetation_mask | lesion_mask

    plant_pixels = int(np.count_nonzero(plant_mask))
    if plant_pixels < 250:
        # Fallback for very close crops where green pixels are limited.
        plant_mask = exg > 2.0
        plant_pixels = int(np.count_nonzero(plant_mask))

    if plant_pixels <= 0:
        return 0.0

    diseased_pixels = int(np.count_nonzero(lesion_mask & plant_mask))
    diseased_percent = (diseased_pixels / plant_pixels) * 100.0
    return round(float(np.clip(diseased_percent, 0.0, 100.0)), 2)


def severity_from_percent(diseased_percent: float) -> str:
    """Convert diseased area percent into readable severity."""
    if diseased_percent <= 1.0:
        return "Minimal"
    if diseased_percent < 10.0:
        return "Low"
    if diseased_percent < 25.0:
        return "Moderate"
    if diseased_percent < 45.0:
        return "High"
    return "Critical"


def get_recovery_suggestions(
    disease_name: str,
    diseased_percent: float,
    confidence: float,
) -> list[str]:
    """Return practical actions after a model diagnosis."""
    if disease_name.startswith("No Disease"):
        return [
            "No active disease pattern detected. Maintain current irrigation and nutrition schedule.",
            "Continue field scouting every 3-4 days and keep leaves dry overnight.",
            "Remove fallen leaf debris from field rows to reduce pathogen carryover.",
        ]

    suggestions = [
        "Isolate visibly infected area and avoid handling healthy plants right after infected ones.",
        "Remove severely affected leaves or branches and dispose away from the field.",
        "Use targeted fungicide/bactericide only after confirming symptoms with a local agronomist.",
        "Avoid overhead irrigation for now; prefer controlled root-zone watering.",
        "Rescan a fresh image in 48-72 hours to track progression.",
    ]

    if "Leaf" in disease_name:
        suggestions.insert(1, "Focus spray coverage on lower and middle leaf canopy where lesions spread fastest.")
    if "Plant" in disease_name:
        suggestions.insert(1, "Inspect stem and nearby plants in the same row to prevent whole-plant spread.")
    if diseased_percent >= 30.0:
        suggestions.append("Affected area is high; prioritize block-level treatment planning immediately.")
    if confidence < 0.65:
        suggestions.append("Prediction confidence is moderate; capture a clearer close-up image for confirmation.")

    return suggestions


def analyze_disease(image_input: Union[str, bytes, bytearray, Image.Image]) -> dict[str, object]:
    """Return full disease analysis for UI: label, confidence, area, severity, suggestions."""
    image = _load_image(image_input)
    disease_name, confidence = _predict_from_loaded_image(image)

    if disease_name.startswith("No Disease"):
        diseased_percent = 0.0
        severity = "Healthy"
    else:
        diseased_percent = estimate_diseased_area_from_image(image)
        severity = severity_from_percent(diseased_percent)

    return {
        "disease_name": disease_name,
        "confidence": confidence,
        "diseased_area_percent": diseased_percent,
        "severity": severity,
        "suggestions": get_recovery_suggestions(disease_name, diseased_percent, confidence),
    }


def predict_disease(image_input: Union[str, bytes, bytearray, Image.Image]) -> tuple[str, float]:
    """Return (disease_name, confidence)."""
    image = _load_image(image_input)
    return _predict_from_loaded_image(image)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python disease_predict.py <image_path>")
        raise SystemExit(1)

    analysis = analyze_disease(sys.argv[1])
    print(
        "Predicted: "
        f"{analysis['disease_name']} "
        f"({float(analysis['confidence']):.3f}), "
        f"Affected: {float(analysis['diseased_area_percent']):.2f}%"
    )
