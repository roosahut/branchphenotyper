# image input output and excel gets and outputs
import os
import pandas as pd
from PIL import Image
from config import augmented_excel_file_path, augmented_sheet_name
import logging

logger = logging.getLogger(__name__)


def get_image(image_path):
    """Loads an image from the specified path."""
    return Image.open(image_path)


def save_image(img_name, img, output_directory, labels):
    """Saves the augmented image and updates the Excel file with corresponding labels."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    full_path = os.path.join(output_directory, img_name)
    img.save(full_path)
    logger.info(f"Augmented image '{img_name}' saved.")
    if labels is None:
        logger.warning(f"Skipping save of {img_name} because labels are missing.")
        return
    update_excel_with_labels(img_name, labels)


def read_excel_labels(image_name, df):
    """Retrieves the label information for a given image name."""
    # Strip augmentation suffixes if present
    base_name = image_name.split("_combo")[0]
    matches = df[df["filename"] == base_name]
    if matches.empty:
        logger.warning(f"No label found for {base_name}")
        return None
    return matches.iloc[0:1]


def update_excel_with_labels(img_name, labels):
    """Updates the Excel sheet with augmented image labels."""
    labels.iloc[:, 0] = img_name
    if os.path.exists(augmented_excel_file_path):
        augmented_excel_df = pd.read_excel(
            augmented_excel_file_path, sheet_name=augmented_sheet_name
        )
        if img_name in augmented_excel_df["filename"].values:
            augmented_excel_df.loc[augmented_excel_df["filename"] == img_name, :] = (
                labels.values[0]
            )
        else:
            augmented_excel_df = pd.concat(
                [augmented_excel_df, labels], ignore_index=True
            )
    else:
        augmented_excel_df = labels
    augmented_excel_df.to_excel(
        augmented_excel_file_path, sheet_name=augmented_sheet_name, index=False
    )


# should we have one function that combines the augmented labels to the original ones?
