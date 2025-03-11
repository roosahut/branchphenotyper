#!/usr/bin/env python3

import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


parser = argparse.ArgumentParser(description="Train a binary classifier for 'orientation' trees using a pretrained model.")
parser.add_argument("--excel_file", type=str, default="/Users/pavlina/Documents/Helsinki/2nd_Semester/DS_project/repo/labels/phenotype_labels.xlsx",
                    help="Path to the Excel file containing columns 'filename' and 'orientation'.")
parser.add_argument("--image_dir", type=str, default = "/Users/pavlina/Documents/Helsinki/2nd_Semester/DS_project/Betula_photos_all",
                    help="Directory where all images are stored.")
parser.add_argument("--output_dir", type=str, default="output_orientation",
                    help="Directory to save the trained model, logs, predictions, etc.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=7,
                    help="Number of epochs for training.")
args = parser.parse_args()


###############################################################################
##################  load data, filter missing ################################
###############################################################################
print("Loading Excel file:", args.excel_file)
df = pd.read_excel(args.excel_file)
# to handle missing
df = df.dropna(subset=["orientation"])

# Convert the label to int to be sure (0/1)
df["orientation"] = df["orientation"].astype(int)

valid_extensions = [".jpg", ".jpeg"]  # i had issues with files while reading, the following just excludes hidden files and such

full_paths = []
valid_labels = []

for i, row in df.iterrows():
    filename = row["filename"]
    label = row["orientation"]
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_extensions:
        continue
    
    img_path = os.path.join(args.image_dir, filename)
    if os.path.isfile(img_path): # just so if there is nonexistent row in excel it does not cause an error
        full_paths.append(img_path)
        valid_labels.append(label)

if not full_paths:
    print("No valid images found. Please check data ir code.")
    sys.exit(1)

full_paths = np.array(full_paths)
valid_labels = np.array(valid_labels)

print(f"Found {len(full_paths)} labeled images for training/validation.")


###############################################################################
##################  # Split into train and validation #########################
###############################################################################
train_paths, val_paths, train_labels, val_labels = train_test_split(
    full_paths, valid_labels, test_size=0.2, random_state=42
)
print(f"Train size: {len(train_paths)}, Validation size: {len(val_paths)}")


###############################################################################
##################  tf.data pipeline ##########################################
###############################################################################
IMG_SIZE = (224, 224)  # apparently typical for many pretrained networks
BATCH_SIZE = args.batch_size

def load_and_preprocess_image(path, label):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  
    return img, label

# Build tf.data Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(load_and_preprocess_image,   num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
train_ds = train_ds.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


###############################################################################
########################## Define custom loss f.  #############################
###############################################################################

# to be revisited
# apparently binary crossentropy might not be ideal and focal loss might be better for unbalanced classes
# but this did not work well, practically only 0s were predited on valiedation set


# def focal_loss(gamma=2.0, alpha=0.25):
#     def loss_fn(y_true, y_pred):
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
#         # Compute cross-entropy loss
#         cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
#         # Compute focal weight
#         focal_weight = alpha * tf.math.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.math.pow(y_pred, gamma) * (1 - y_true)
#         loss = focal_weight * cross_entropy
#         return tf.reduce_mean(loss)
#     return loss_fn


###############################################################################
########################## Model from backbone  ###############################
###############################################################################
backbone = tf.keras.applications.EfficientNetV2B3(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg", 
    weights="imagenet"
)

# Freeze the backbone
for layer in backbone.layers:
    layer.trainable = False

model = tf.keras.Sequential([
    # Data augmentation
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    backbone,

    # Additional layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()


###############################################################################
########################## Train the model  ###################################
###############################################################################
os.makedirs(args.output_dir, exist_ok=True)

# this I use because classes are unbalanced
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    class_weight=class_weights 
)


# fine tune backbone
for layer in backbone.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name='recall')]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights
)


###############################################################################
################### Model saving  #############################################
###############################################################################
model_save_path = os.path.join(args.output_dir, "orientation_model.keras")
model.save(model_save_path)
print("Model saved to:", model_save_path)


###############################################################################
################### Validation predictions  ###################################
###############################################################################
val_labels_pred = model.predict(val_ds)  # predictions in [0,1]
val_labels_pred = (val_labels_pred >= 0.5).astype(int).flatten() # maybe we could try different p values?

predictions_save_path = os.path.join(args.output_dir, "validation_predictions.csv")
val_filenames = val_paths
df_preds = pd.DataFrame({
    "filename": val_filenames,
    "y_true": val_labels,
    "y_pred": val_labels_pred
})
df_preds.to_csv(predictions_save_path, index=False)
print("Validation predictions saved to:", predictions_save_path)


###############################################################################
################### Ploting training curves  ##################################
###############################################################################
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(args.output_dir, "loss.png")
plt.savefig(loss_plot_path)
plt.show()
print("Saved loss plot to:", loss_plot_path)

# accuracy plot
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
acc_plot_path = os.path.join(args.output_dir, "accuracy.png")
plt.savefig(acc_plot_path)
plt.show()
print("Saved accuracy plot to:", acc_plot_path)