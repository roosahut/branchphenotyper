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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# arguments
parser = argparse.ArgumentParser(description="Train an ordinal classifier for 'weeping' using a pretrained model.")
parser.add_argument("--excel_file", type=str, default="/Users/pavlina/Documents/Helsinki/2nd_Semester/DS_project/repo/labels/phenotype_labels.xlsx",
                    help="Path to the Excel file containing columns 'filename' and 'weeping' [0..5].")
parser.add_argument("--image_dir", type=str, default = "/Users/pavlina/Documents/Helsinki/2nd_Semester/DS_project/Betula_photos_all",
                    help="Directory where all images are stored.")
parser.add_argument("--output_dir", type=str, default="output_weeping",
                    help="Directory to save the trained model, logs, predictions, etc.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=12,
                    help="Number of epochs for training.")
args = parser.parse_args()


###############################################################################
##################  load data, filter missing ################################
###############################################################################
print("Loading Excel file:", args.excel_file)
df = pd.read_excel(args.excel_file)

df = df.dropna(subset=["weeping"])
df["weeping"] = df["weeping"].astype(int)

valid_extensions = [".jpg", ".jpeg"]
full_paths = []
valid_labels = []

for i, row in df.iterrows():
    filename = row["filename"]
    label = row["weeping"]  # integer in [0, 1, 2, 3, 4, 5]

    ext = os.path.splitext(filename)[1].lower() # consider just photos
    if ext not in valid_extensions:
        continue

    img_path = os.path.join(args.image_dir, filename)
    if os.path.isfile(img_path):
        full_paths.append(img_path)
        valid_labels.append(label)

if not full_paths:
    print("No valid images found. Please check data ir code.")
    sys.exit(1)

full_paths = np.array(full_paths)
valid_labels = np.array(valid_labels)
print(f"Found {len(full_paths)} labeled images for training/validation.")


###############################################################################
################## ORDINAL LABELLING ##########################################
###############################################################################

# i found this approach in stack overflow but maybe different approach could work better?
# haven't yet trained NN with ordinal data so...
def encode_ordinal(labels, num_classes=6):
    """
    labels: array of shape (N,) with integer classes in [0..num_classes-1]
    returns: array of shape (N, num_classes-1)
    """
    # if we have 6 classes [0..5], then we create 5 ">= k" thresholds
    # for k in [1..5].
    encoded = np.zeros((len(labels), num_classes - 1), dtype=np.int32)
    for i, c in enumerate(labels):
        # eg: c=3 -> [1,1,1,0,0] -> means it's >=1, >=2, >=3, not >=4, not >=5.
        encoded[i, :c] = 1
    return encoded

ordinal_labels = encode_ordinal(valid_labels, num_classes=6)


###############################################################################
##################  # Split into train and validation #########################
###############################################################################
train_paths, val_paths, train_labels, val_labels = train_test_split(
    full_paths, ordinal_labels, test_size=0.2, random_state=42
)
print(f"Train size: {len(train_paths)}, Validation size: {len(val_paths)}")


###############################################################################
##################  tf.data pipeline ##########################################
###############################################################################
IMG_SIZE = (224, 224)
BATCH_SIZE = args.batch_size

def load_and_preprocess_image(path, label):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds   = tf.data.Dataset.from_tensor_slices((val_paths,   val_labels))

train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(load_and_preprocess_image,   num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


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
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    backbone,

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    #5 outputs (for classes [0..5]) with sigmoid
    tf.keras.layers.Dense(5, activation='sigmoid')
])

#binary_crossentropy because each of the 5 outputs is a separate binary target.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["mae", "mse"]
)

model.summary()


###############################################################################
########################## Train the model  ###################################
###############################################################################
os.makedirs(args.output_dir, exist_ok=True)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(valid_labels),
    y=valid_labels.flatten()
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    class_weight=class_weights
)


# here, finetuning made it actually worse for me but maybe i'm doing it wrong 
# # for layer in backbone.layers[-20:]:
# for layer in backbone.layers[-10:]:
#     layer.trainable = True

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss="binary_crossentropy",
#     metrics=["accuracy", tf.keras.metrics.Recall(name='recall')]
# )

# # Fine-tune for a few additional epochs
# history_finetune = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=3,
#     class_weight=class_weights
# )


###############################################################################
################### Model saving  #############################################
###############################################################################
model_save_path = os.path.join(args.output_dir, "weeping_model.keras")
model.save(model_save_path)
print("Model saved to:", model_save_path)


###############################################################################
################### Validation predictions  ###################################
###############################################################################
val_preds = model.predict(val_ds)  # shape (N, 5)
val_preds_binary = (val_preds >= 0.5).astype(int)  # shape (N, 5)
val_preds_class = np.sum(val_preds_binary, axis=1)  # back to [0..5]

# val_labels shape is (N, 5). decoding them to original 0..5
def decode_ordinal(ordinal_labels):
    """
    ordinal_labels: (N, 5) with 0 or 1
    returns: array of shape (N,) with integer classes in [0..5]
    """
    return np.sum(ordinal_labels, axis=1)

val_labels_class = decode_ordinal(val_labels)

# predictions
predictions_save_path = os.path.join(args.output_dir, "validation_predictions.csv")
val_filenames = val_paths
df_preds = pd.DataFrame({
    "filename": val_filenames,
    "y_true": val_labels_class,
    "y_pred": val_preds_class
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

plt.figure(figsize=(8,5))
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.title("MAE over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
acc_plot_path = os.path.join(args.output_dir, "mae.png")
plt.savefig(acc_plot_path)
plt.show()
print("Saved accuracy plot to:", acc_plot_path)

plt.figure(figsize=(8,5))
plt.plot(history.history["mse"], label="Train MSE")
plt.plot(history.history["val_mse"], label="Val MSE")
plt.title("MSE over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
acc_plot_path = os.path.join(args.output_dir, "mse.png")
plt.savefig(acc_plot_path)
plt.show()
print("Saved accuracy plot to:", acc_plot_path)