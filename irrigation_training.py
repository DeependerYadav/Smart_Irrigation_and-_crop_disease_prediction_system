"""Train irrigation neural-network regression model.

Model input features:
- soil_moisture
- temperature
- humidity
- rain_forecast
- crop_type

Target:
- water_required (liters)

Artifacts:
- irrigation_model.h5
- scaler.pkl
"""

from pathlib import Path
import argparse
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "irrigation_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rain_forecast",
    "crop_type",
]
TARGET_COLUMN = "water_required"


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic irrigation dataset."""
    rng = np.random.default_rng(random_state)

    soil_moisture = rng.uniform(10, 80, n_samples)
    temperature = rng.uniform(20, 40, n_samples)
    humidity = rng.uniform(30, 90, n_samples)
    rain_forecast = rng.integers(0, 2, n_samples)
    crop_type = rng.integers(0, 5, n_samples)

    # Synthetic target formula.
    water_required = (
        120
        - soil_moisture
        + 0.6 * temperature
        - 0.4 * humidity
        - 20 * rain_forecast
        + 5 * crop_type
    )

    return pd.DataFrame(
        {
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rain_forecast": rain_forecast,
            "crop_type": crop_type,
            "water_required": water_required,
        }
    )


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load optional real dataset and validate columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Required: {required}")

    df = df[required].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset has no valid rows after removing missing values.")
    return df


def build_model(input_dim: int) -> tf.keras.Model:
    """Build a dense neural network for regression."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_save(
    df: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    test_size: float,
    random_state: int,
    epochs: int,
    batch_size: int,
) -> None:
    """Split data, train model, evaluate, and save artifacts."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(input_dim=X_train_scaled.shape[1])
    model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    y_pred = model.predict(X_test_scaled, verbose=0).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train irrigation neural network model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Optional CSV path with columns: "
            "soil_moisture, temperature, humidity, rain_forecast, crop_type, water_required"
        ),
    )
    parser.add_argument("--samples", type=int, default=1000, help="Synthetic sample count")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-model", type=str, default=str(MODEL_PATH), help="Output .h5 model path")
    parser.add_argument("--output-scaler", type=str, default=str(SCALER_PATH), help="Output scaler .pkl path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)

    if args.data:
        print(f"Loading dataset from: {args.data}")
        dataset = load_dataset(Path(args.data))
    else:
        print("No dataset provided. Generating synthetic irrigation data...")
        dataset = generate_synthetic_data(
            n_samples=args.samples,
            random_state=args.random_state,
        )

    train_and_save(
        df=dataset,
        model_path=Path(args.output_model),
        scaler_path=Path(args.output_scaler),
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
