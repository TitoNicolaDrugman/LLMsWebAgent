import os
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & HYPERPARAMETERS ---

# Use the full aspect ratio, but maybe scale down slightly to save VRAM/Speed
# Original: 1006 x 620. Let's try 50% scale which is plenty for readability.
IMG_WIDTH = 503  
IMG_HEIGHT = 310
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) # (Height, Width, Channels)

BATCH_SIZE_PER_GPU = 16 # Adjust based on VRAM. 16 * 2 GPUs = 32 global batch size
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * 2 

EPOCHS_HEAD = 10      # Phase 1: Train only classifier head
EPOCHS_FINE_TUNE = 40 # Phase 2: Train whole model (slowly)

CSV_FILE_PATH = "master_dataset_v6.csv"
MODEL_SAVE_PATH = "best_webpage_classifier_v2.keras"

# Websites to completely exclude from training (Hold-out test set)
TEST_WEBSITES = ["GitHub", "Booking"] 
# Websites to ignore (Noise/Errors)
EXCLUDE_WEBSITES = ["BBC News", "Google Flights", "Google Map", "Google Search", "Wolfram Alpha", "Huggingface"]

# --- 2. MULTI-GPU SETUP ---
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# --- 3. DATA PREPARATION ---
def load_data():
    print("--- Loading Data ---")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Filter Errors
    error_labels = ['MODEL_ERROR', 'IMAGE_READ_ERROR', 'OTHER', 'Unknown']
    df = df[~df['label'].isin(error_labels)]
    
    # Extract Website Name
    def extract_website(task_id):
        match = re.search(r'task(.*?)(--\d+)', str(task_id))
        return match.group(1) if match else "Unknown"
    df['website'] = df['task_id'].apply(extract_website)
    
    # Filter Excluded Sites
    df = df[~df['website'].isin(EXCLUDE_WEBSITES)]
    
    # Split Hold-out Test Set (Domain Generalization Test)
    test_df = df[df['website'].isin(TEST_WEBSITES)].copy()
    train_val_df = df[~df['website'].isin(TEST_WEBSITES)].copy()
    
    # Split Train/Val by TASK ID (Prevent data leakage)
    unique_tasks = train_val_df['task_id'].unique()
    train_tasks, val_tasks = train_test_split(unique_tasks, test_size=0.15, random_state=42)
    
    train_df = train_val_df[train_val_df['task_id'].isin(train_tasks)]
    val_df = train_val_df[train_val_df['task_id'].isin(val_tasks)]
    
    print(f"Train Imgs: {len(train_df)} | Val Imgs: {len(val_df)} | Test Imgs: {len(test_df)}")
    return train_df, val_df, test_df

train_df, val_df, test_df = load_data()
classes = sorted(train_df['label'].unique())
num_classes = len(classes)
print(f"Classes ({num_classes}): {classes}")

# --- 4. IMAGE GENERATORS & AUGMENTATION ---
# Crucial: Webpages contain text. 
# DO NOT use horizontal_flip=True (Mirroring text makes it unreadable/confusing).
# DO NOT use heavy rotation (Webpages are rectangular).

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input, # Handles 0-255 to -1..1 or 0..1
    width_shift_range=0.05,  # Slight shift
    height_shift_range=0.05, # Slight shift
    zoom_range=0.05,         # Slight zoom
    brightness_range=[0.9, 1.1], # Slight lighting change
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=GLOBAL_BATCH_SIZE,
    class_mode='categorical', classes=classes, shuffle=True
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=GLOBAL_BATCH_SIZE,
    class_mode='categorical', classes=classes, shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df, x_col='image_path', y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=GLOBAL_BATCH_SIZE,
    class_mode='categorical', classes=classes, shuffle=False
)

# --- 5. CLASS WEIGHTS (HANDLE IMBALANCE) ---
# Calculate weights so "Loading State" (rare) counts as much as "Popup" (common)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(weights))
print("\nComputed Class Weights (Higher = Rare Class):")
for idx, weight in class_weights.items():
    print(f"  {classes[idx]}: {weight:.2f}")

# --- 6. MODEL BUILDING (INSIDE STRATEGY SCOPE) ---
with strategy.scope():
    # Load EfficientNetV2S (Better than B0, faster/better params)
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # FREEZE the base model initially
    base_model.trainable = False
    
    # Create the Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) # Stabilize training
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # Compilation
    # Use standard CrossEntropy. 
    # Label Smoothing helps preventing the model from being "too confident" on noisy data
    optimizer = optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'AUC'])

model.summary()

# --- 7. TRAINING ---

# Callbacks
checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_AUC', save_best_only=True, mode='max', verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_AUC', patience=8, mode='max', restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

print("\n=== PHASE 1: Training Head (Base Frozen) ===")
history_phase1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr]
)

print("\n=== PHASE 2: Fine-Tuning (Unfreezing Top Layers) ===")
with strategy.scope():
    # Unfreeze the top 30 layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Recompile with a MUCH lower learning rate to nudge weights gently
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5), # 100x smaller LR
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'AUC']
    )

history_phase2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# --- 8. EVALUATION ---
print("\n=== FINAL EVALUATION ON TEST SET (GitHub / Booking) ===")
# Load best model
best_model = models.load_model(MODEL_SAVE_PATH)

# Predictions
y_true = test_gen.classes
y_pred_prob = best_model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)

# Report
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix on Hold-Out Test Set')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")