"""
Sports Image Classification - Phase 2 Code Implementation
Dataset: https://www.kaggle.com/datasets/gpiosenka/sports-classification
Models: Custom CNN and MobileNetV2 Transfer Learning
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_CNN = 15
EPOCHS_TRANSFER = 10

# Kaggle usually stores this dataset under /kaggle/input/sports-classification
# The actual image folders are commonly inside a subfolder named "sports".
POSSIBLE_DATASET_DIRS = [
    "/kaggle/input/sports-classification/sports",
    "/kaggle/input/sports-classification",
    "./sports",
    "./dataset/sports",
    "./dataset"
]

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def find_dataset_dir():
    """Find dataset directory containing train, valid, and test folders."""
    for path in POSSIBLE_DATASET_DIRS:
        if all(os.path.isdir(os.path.join(path, split)) for split in ["train", "valid", "test"]):
            return path
    raise FileNotFoundError(
        "Dataset directory not found. Please update POSSIBLE_DATASET_DIRS so it points to the folder containing train, valid, and test."
    )


DATASET_DIR = find_dataset_dir()
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

print("Dataset directory:", DATASET_DIR)

# -----------------------------
# Dataset Loading and Preprocessing
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    seed=SEED,
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Number of classes:", NUM_CLASSES)

with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)

# Improve pipeline speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Data augmentation for training robustness
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# -----------------------------
# Helper Functions
# -----------------------------
def plot_training_curves(history, title_prefix, filename):
    """Save accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def get_predictions(model, dataset):
    """Return true labels and predicted labels."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


def evaluate_model(model, dataset, model_name):
    """Evaluate model and save confusion matrix/classification report."""
    y_true, y_pred = get_predictions(model, dataset)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

# -----------------------------
# Custom CNN Model Development
# -----------------------------
def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="Custom_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


cnn_model = build_custom_cnn()
cnn_model.summary()

callbacks_cnn = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, "custom_cnn_model.keras"), save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
]

history_cnn = cnn_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_CNN,
    callbacks=callbacks_cnn
)

plot_training_curves(history_cnn, "Custom CNN", "cnn_accuracy_loss_curves.png")
cnn_results = evaluate_model(cnn_model, test_ds, "cnn")

# -----------------------------
# Transfer Learning Model: MobileNetV2
# -----------------------------
def build_mobilenetv2_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Transfer_Learning")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


mobilenet_model = build_mobilenetv2_model()
mobilenet_model.summary()

callbacks_transfer = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, "mobilenetv2_model.keras"), save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
]

history_mobilenet = mobilenet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_TRANSFER,
    callbacks=callbacks_transfer
)

plot_training_curves(history_mobilenet, "MobileNetV2", "mobilenet_accuracy_loss_curves.png")
mobilenet_results = evaluate_model(mobilenet_model, test_ds, "mobilenet")

# -----------------------------
# Model Comparison Table
# -----------------------------
comparison_df = pd.DataFrame([cnn_results, mobilenet_results])
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\nModel Comparison:")
print(comparison_df)

# Save final models
cnn_model.save(os.path.join(MODEL_DIR, "custom_cnn_final.keras"))
mobilenet_model.save(os.path.join(MODEL_DIR, "mobilenetv2_final.keras"))

print("\nTraining complete. Figures, reports, comparison table, and models saved.")
