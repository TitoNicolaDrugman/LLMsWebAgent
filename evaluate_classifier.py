# evaluate_classifier.py

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIG (must match train_classifier_new.py) ---

IMG_WIDTH = 503
IMG_HEIGHT = 310
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

CSV_FILE_PATH = "master_dataset_v6.csv"
MODEL_SAVE_PATH = "best_webpage_classifier_v2.keras"

TEST_WEBSITES = ["GitHub", "Booking"]
EXCLUDE_WEBSITES = ["BBC News", "Google Flights", "Google Map",
                    "Google Search", "Wolfram Alpha", "Huggingface"]

# --- 2. LOAD DATA & BUILD TEST SPLIT ---

def extract_website(task_id):
    match = re.search(r"task(.*?)(--\d+)", str(task_id))
    return match.group(1) if match else "Unknown"

print("--- Loading Data for Evaluation ---")
df = pd.read_csv(CSV_FILE_PATH)

# Remove error rows exactly like training
error_labels = ["MODEL_ERROR", "IMAGE_READ_ERROR", "OTHER", "Unknown"]
df = df[~df["label"].isin(error_labels)]

# Website column
df["website"] = df["task_id"].apply(extract_website)

# Drop excluded websites
df = df[~df["website"].isin(EXCLUDE_WEBSITES)]

# Hold‑out test websites
test_df = df[df["website"].isin(TEST_WEBSITES)].copy()

print(f"Test images: {len(test_df)}")

# Classes are defined from the TRAINING domain (same ordering as before)
train_domain_df = df[~df["website"].isin(TEST_WEBSITES)].copy()
classes = sorted(train_domain_df["label"].unique())
num_classes = len(classes)
print(f"Classes ({num_classes}): {classes}")

# --- 3. TEST GENERATOR (same preprocessing as training) ---

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode="categorical",
    classes=classes,
    shuffle=False,
)

# --- 4. LOAD BEST MODEL AND PREDICT ---

print("\nLoading best model...")
best_model = models.load_model(MODEL_SAVE_PATH)

print("Running predictions on test set...")
y_true = test_gen.classes
y_pred_prob = best_model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)

# --- 5. CLASSIFICATION REPORT LIMITED TO PRESENT CLASSES ---

present_labels = np.unique(y_true)
present_class_names = [classes[i] for i in present_labels]

print("\n=== Classification Report (GitHub + Booking) ===")
print(
    classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_class_names,
        digits=3,
    )
)

# --- 6. CONFUSION MATRIX (OPTIONAL) ---

cm = confusion_matrix(y_true, y_pred, labels=present_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=present_class_names,
    yticklabels=present_class_names,
)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix on GitHub/Booking Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_test_only.png")
print("\nConfusion matrix saved to confusion_matrix_test_only.png")
