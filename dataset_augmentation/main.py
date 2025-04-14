"""
Main entry point to run the augmentation pipelines.
Choose between full dataset balancing (recommended), rare-only, individual image testing, or demo singular transform.
"""

from config import (
    image_directory,
    output_directory,
    excel_file_path,
    sheet_name,
    image_filename,
    os,
    target_count_strategy,
    target_value,
    max_extra_aug_per_image,
    num_augmented_images,
    augmented_excel_file_path,
    augmented_sheet_name,
)
from augmentation_pipeline import (
    augmentation_for_image,
    combined_augmentation_to_all_imgs_with_balancing,
)
from file_handling import get_image
from transformations import rotate_image
import pandas as pd
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("label_analysis_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def run_combined_augmentation_with_balancing():
    """
    Creates new images by augmentation from the full dataset and adds creates additional images by augmentation
    for underrepresented label combinations, using dynamic rarity balancing.

    This function:
    - Loads label data from an Excel file and filters it to include only images that physically exist
    - Applies `num_augmented_images` augmentations to **every image**, using randomly combined transformations
    - Identifies **rare label combinations** using a selected strategy (`fixed`, `percentile`, or `max`)
    - Augments rare images further, based on how far they fall below the `target_value`, up to `max_extra_aug_per_image`
    - Logs progress and results to `label_analysis_log.txt`
    - Saves all augmented images and updates their label records in a separate Excel file

    Parameters are taken from `config.py` and include:
    - Paths and names of input and output folders and excel: `image_directory`, `output_directory`, `excel_file_path`, `sheet_name`
    - `num_augmented_images`: Number of augmented versions to create per image regardless of rarity
    - `target_count_strategy`: Rarity detection strategy ('fixed', 'percentile', or 'max')
    - `target_value`: Numeric threshold for rarity
    - `max_extra_aug_per_image`: Limit for how many extra augmentations to apply
    """

    logger.info(
        f"Running combined augmentation (chaining random transformations) on the entire dataset and creating extra augmented images for rare label types..."
    )

    combined_augmentation_to_all_imgs_with_balancing(
        image_directory,
        output_directory,
        excel_file_path,
        sheet_name,
        augmented_excel_file_path,
        augmented_sheet_name,
        num_augmented_images,
        target_count_strategy,
        target_value,
        max_extra_aug_per_image,
    )
    logger.info("Dataset expansion and balancing completed successfully.")


def run_all_augmentations_for_one_image():
    """
    Runs all predefined individual augmentation functions on a single image.

    This function loads the Excel label information, applies all individual augmentations
    to the image (each resulting in a separate saved image), and updates the Excel file accordingly.
    Mostly for testing and exploration

    Parameters:
        image_path (str): Full path to the image.
        image_filename (str): The filename of the image.
    """
    image_path = image_directory + "/" + image_filename
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    print(f"Running all individual augmentations on image: {image_filename}")
    augmentation_for_image(image_path, output_directory, image_filename, df)


def run_one_augmentation_for_one_image():
    """
    Applies a single transformation (rotate) to a chosen image for exploration purposes.
    This function loads an image, applies the 'rotate' transformation, and displays the result.
    This functions does NOT read or modify any excel files.
    Mostly for testing and exploration
    """
    print("\nApplying a single transformation for exploration purposes...")

    image = get_image(image_path="./images/original_images/B-1.jpg")

    transformed_image = rotate_image(image)
    transformed_image.show()  # This will open the image in your default image viewer


def delete_augmented_images_folder_contents():
    """Deletes all files in the augmented images output directory"""
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        os.remove(file_path)


# RUN PIPELINES FROM HERE: UNCOMMENT WHICH PIPELINE YOU WANT TO RUN
# put necessary parameters into config.py file
# you will most likely want to run function: run_combined_augmentation_with_balancing()


start_time = time.time()

run_combined_augmentation_with_balancing()
# run_all_augmentations_for_one_image()
# run_one_augmentation_for_one_image()

end_time = time.time()
duration = end_time - start_time

print(f"pipeline ran for {duration:.2f} seconds or {duration/60:.2f} minutes")

# delete_augmented_images_folder_contents()
