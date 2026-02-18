"""Train cotton disease classification model (cotton-only).

Uses the reference notebook pattern:
- EfficientNetB0 transfer learning backbone
- 224x224 image size
- Data augmentation
- Train/val/test folders from Cotton Disease dataset
"""

from pathlib import Path
import argparse
import pickle

import tensorflow as tf


BASE_DIR = Path(__file__).parent
DEFAULT_DATA_ROOT = Path(r"C:\Users\HP\Downloads\data for cotton\Cotton Disease")
MODEL_PATH = BASE_DIR / "disease_model.h5"
LABELS_PATH = BASE_DIR / "labels.pkl"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def resolve_data_dirs(data_root: Path) -> tuple[Path, Path, Path]:
    """Return train/val/test directories and validate they exist."""
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    missing = [p for p in (train_dir, val_dir, test_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset folders not found. Expected: "
            f"{train_dir}, {val_dir}, {test_dir}"
        )
    return train_dir, val_dir, test_dir


def load_split_dataset(directory: Path, shuffle: bool) -> tf.data.Dataset:
    """Load one dataset split from a directory."""
    return tf.keras.preprocessing.image_dataset_from_directory(
        str(directory),
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def load_datasets(data_root: Path):
    """Load train/val/test datasets and class names."""
    train_dir, val_dir, test_dir = resolve_data_dirs(data_root)

    print(f"Loading train data from: {train_dir}")
    train_ds = load_split_dataset(train_dir, shuffle=True)
    print(f"Loading val data from: {val_dir}")
    val_ds = load_split_dataset(val_dir, shuffle=False)
    print(f"Loading test data from: {test_dir}")
    test_ds = load_split_dataset(test_dir, shuffle=False)

    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError(
            "Class mismatch between train/val/test folders. "
            "Make sure each split has identical class subfolders."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int) -> tuple[tf.keras.Model, str]:
    """Build transfer learning model with robust pretrained fallback."""
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomHeight(0.2),
            tf.keras.layers.RandomWidth(0.2),
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs, training=True)

    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )
        backbone_name = "EfficientNetB0 (ImageNet)"
    except Exception as efficientnet_exc:
        print(f"Warning: EfficientNet ImageNet weights failed: {efficientnet_exc}")
        print("Trying MobileNetV2 ImageNet weights as fallback...")
        try:
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            )
            backbone_name = "MobileNetV2 (ImageNet fallback)"
        except Exception as mobilenet_exc:
            print(f"Warning: MobileNetV2 ImageNet weights also failed: {mobilenet_exc}")
            print("Falling back to EfficientNetB0 with random initialization.")
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            )
            backbone_name = "EfficientNetB0 (random init)"

    base_model._name = "feature_extractor"
    base_model.trainable = False
    # Keep preprocessing serializable in .h5 format.
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale_m1p1")(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="cotton_disease_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, backbone_name


def fine_tune_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    fine_tune_epochs: int,
    initial_epochs_done: int,
) -> tf.keras.callbacks.History | None:
    """Unfreeze top layers and continue training for better accuracy."""
    if fine_tune_epochs <= 0:
        return None

    try:
        base_model = model.get_layer("feature_extractor")
    except Exception:
        return None

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print(f"Fine-tuning for {fine_tune_epochs} epochs...")
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs_done + fine_tune_epochs,
        initial_epoch=initial_epochs_done,
    )


def train_and_save(
    data_root: Path,
    model_path: Path,
    labels_path: Path,
    epochs: int,
    fine_tune_epochs: int,
) -> None:
    """Train cotton model and save artifacts."""
    train_ds, val_ds, test_ds, class_names = load_datasets(data_root)
    print(f"Classes found: {class_names}")

    model, backbone_name = build_model(num_classes=len(class_names))
    print(f"Backbone in use: {backbone_name}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print(f"Initial training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    fine_tune_history = fine_tune_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        fine_tune_epochs=fine_tune_epochs,
        initial_epochs_done=len(history.history.get("loss", [])),
    )

    print("Evaluating on test split...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    model.save(model_path)
    with open(labels_path, "wb") as file_obj:
        pickle.dump(class_names, file_obj)

    print(f"Model saved to: {model_path}")
    print(f"Labels saved to: {labels_path}")

    if fine_tune_history is not None:
        last_val = fine_tune_history.history.get("val_accuracy", [None])[-1]
    else:
        last_val = history.history.get("val_accuracy", [None])[-1]
    if last_val is not None:
        print(f"Final validation accuracy: {last_val:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cotton disease classifier")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to Cotton Disease dataset root (containing train/val/test)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(MODEL_PATH),
        help="Output path for model (.h5)",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default=str(LABELS_PATH),
        help="Output path for labels pickle (.pkl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Initial training epochs",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=2,
        help="Fine-tuning epochs after initial training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_root=Path(args.data_root),
        model_path=Path(args.output_model),
        labels_path=Path(args.output_labels),
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )
