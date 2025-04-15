"""
This module contains functions for applying image augmentations at both the
single-image and dataset scale. It supports two types of augmentations:
- Individual augmentations: one transformation per output image
- Combined augmentations: multiple transformations chained per output image

Includes logic for augmenting:
- Every image in the dataset
- Only rare label combinations (as determined by a rarity strategy)

All outputs are saved to disk and label tracking is automatically updated.
"""

import os
import pandas as pd
import random
import os.path
from os.path import splitext
from file_handling import get_image, save_image, read_excel_labels
from dataset_selection import (
    determine_target_count,
    get_augmentation_plan_by_combination_balanced,
    log_dataset_summary,
    get_individual_label_augmentation_plan
)
from transformations import (
    flip_image,
    zoom_in,
    rotate_image,
    increase_brightness,
    increase_contrast,
    adjust_saturation,
    blur_image,
    warp_perspective,
    rgb_shift,
    channel_shuffle,
    convert_to_grayscale,
)
import logging

logger = logging.getLogger(__name__)

# FOR ONE IMAGE:


def augmentation_for_image(image_path, output_directory, img_filename, df):
    """
    Applies all individual augmentation functions to one image.

    Each transformation (e.g. rotate, flip, grayscale) is applied separately,
    generating one new image per transformation.

    Parameters:
        image_path (str): Path to the original image file.
        output_directory (str): Directory where augmented images are saved.
        img_filename (str): Filename of the original image.
        df (pd.DataFrame): DataFrame containing the labels for all images.

    Returns:
        None. Augmented images are saved, and labels are recorded.
    """
    img = get_image(image_path)
    labels = read_excel_labels(img_filename, df)
    if labels is None:
        logger.warning(f"Skipping {img_filename} due to missing label.")
        return
    transformations = {
        "flipped": flip_image(img),
        "rotated": rotate_image(img),
        "zoomed_in": zoom_in(img),
        "bright": increase_brightness(img),
        "contrast": increase_contrast(img),
        "saturated": adjust_saturation(img),
        "blurred": blur_image(img),
        "perspective": warp_perspective(img),
        "rgb_shift": rgb_shift(img),
        "channel_shuffle": channel_shuffle(img),
        "grayscale": convert_to_grayscale(img),
    }

    base_name, ext = splitext(img_filename)
    for suffix, transformed_img in transformations.items():
        new_filename = f"{base_name}_{suffix}{ext}"
        save_image(new_filename, transformed_img, output_directory, labels)


def combined_augmentation_for_image(
    image_path, output_directory, img_filename, df, num_augmented_images
):
    """
    Applies random combinations of augmentations to a single image.
    Repeats the process `num_augmented_images` times.

    For each of `num_augmented_images`, a random subset of transformations is
    selected and applied sequentially. Transformation names are encoded in the
    output filename. Minimum of two transformations done.

    Parameters:
        image_path (str): Path to the image being augmented.
        output_directory (str): Directory to save new images.
        img_filename (str): Original image filename.
        df (pd.DataFrame): DataFrame with label metadata.
        num_augmented_images (int): Number of new augmented versions to generate.

    Returns:
        None. Saves new images and logs label tracking updates.
    """
    img = get_image(image_path)
    labels = read_excel_labels(img_filename, df)
    if labels is None:
        logger.warning(f"Skipping {img_filename} due to missing label.")
        return

    available_transformations = {
        "flipped": flip_image,
        "rotated": rotate_image,
        "zoomed_in": zoom_in,
        "bright": increase_brightness,
        "contrast": increase_contrast,
        "saturated": adjust_saturation,
        "blurred": blur_image,
        "perspective": warp_perspective,
        "rgb_shift": rgb_shift,
        "channel_shuffle": channel_shuffle,
        "grayscale": convert_to_grayscale,
    }

    base_name, ext = splitext(img_filename)
    for i in range(num_augmented_images):
        num_transformations = random.randint(2, len(available_transformations))
        transformation_keys = random.sample(
            list(available_transformations.keys()), num_transformations
        )
        augmented_img = img.copy()
        applied_transformations = []
        for key in transformation_keys:
            transformation_func = available_transformations[key]
            augmented_img = transformation_func(augmented_img)
            applied_transformations.append(key)
        transformation_str = "_".join(applied_transformations)
        new_filename = f"{base_name}_combo{i+1}_{transformation_str}{ext}"
        save_image(new_filename, augmented_img, output_directory, labels)


# FOR MULTIPLE IMAGES:


def combined_augmentation_to_all_imgs_with_balancing(
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
):
    """
    Two-phase augmentation pipeline:

    PHASE 1 (dataset expansion):
        - Applies `num_augmented_images` augmentations to every original image in the dataset.
        - Expands dataset while preserving label distribution.

    PHASE 2 (dataset balancing):
        - Reloads updated label data after baseline.
        - Computes actual label combo counts.
        - Finds underrepresented combos (< target_count).
        - Matches those combos back to original images only.
        - Distributes extra augmentations evenly across originals.

    Parameters:
        image_directory (str): Path to original images.
        output_directory (str): Where to save augmented outputs.
        excel_file_path (str): Path to Excel label file.
        sheet_name (str): Sheet containing label data.
        num_augmented_images (int): Number of baseline augmentations per image.
        target_count_strategy (str): 'fixed', 'percentile', or 'max'.
        target_value (int or float): Used with the rarity strategy.
        max_extra_aug_per_image (int): Cap per image for rare balancing.

    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    logger.info(
        "PHASE 1: Starting baseline augmentations for all images to increase dataset size..."
    )

    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    existing_images = set(os.listdir(image_directory))
    df = df[df["filename"].isin(existing_images)].copy()
    label_columns = [col for col in df.columns if col != "filename"]

    logger.info(f"Found {len(existing_images)} images matching excel filenames")
    log_dataset_summary(df, label_columns, "Original")

    logger.info(f"Generating {num_augmented_images} new images from all images.")
    for img_filename in df["filename"]:
        img_path = os.path.join(image_directory, img_filename)
        combined_augmentation_for_image(
            img_path,
            output_directory,
            img_filename,
            df,
            num_augmented_images=num_augmented_images,
        )

    logger.info("Baseline augmentation complete.")

    logger.info("Reloading augmented label file after baseline.")
    df_aug = pd.read_excel(augmented_excel_file_path, sheet_name=augmented_sheet_name)

    log_dataset_summary(df_aug, label_columns, "Post-baseline")

    logger.info("Calculating rarity and balancing targets with the given strategy...")
    target_count = determine_target_count(
        df_aug,
        label_columns,
        method=target_count_strategy,
        value=target_value,
        baseline_augments_per_image=0,  # already included in df_aug
    )
    logger.info(f"Target per label combination: {target_count}")

    logger.info(
        "Calculating how many new images each rare label combination needs to be generated to balance label distribution"
    )
    plan_df = get_augmentation_plan_by_combination_balanced(
        df_original=df,
        df_augmented=df_aug,
        label_columns=label_columns,
        target_count=target_count,
        max_extra_aug_per_image=max_extra_aug_per_image,
    )
    logger.info(f"Final rare image augmentation plan ({len(plan_df)} rows):")
    logger.info("\n%s", plan_df.to_string(index=False))

    if plan_df.empty:
        logger.info(
            "No extra augmentations needed. All label combinations are already balanced."
        )
        return

    logger.info("PHASE 2: Applying extra augmentations to rare originals...")
    for row in plan_df.itertuples():
        logger.info(
            f"Augmenting original image: {row.filename} ({row.augmentations_needed} extra)"
        )
        img_path = os.path.join(image_directory, row.filename)

        for _ in range(row.augmentations_needed):
            combined_augmentation_for_image(
                img_path,
                output_directory,
                row.filename,
                df,  # original so the writing to aug excel works
                num_augmented_images=1,
            )

    # Reload and log final result
    final_df = pd.read_excel(augmented_excel_file_path, sheet_name=augmented_sheet_name)

    log_dataset_summary(final_df, label_columns, "Final after balancing")

    original_count = df["filename"].nunique()
    final_count = final_df["filename"].nunique()
    new_images_created = final_count - original_count

    logger.info(f"Total new images created: {new_images_created}")

    growth_factor = final_count / original_count if original_count > 0 else 0
    logger.info(f"Dataset growth: {growth_factor:.2f}x")

    combo_counts_final = (
        final_df[label_columns].value_counts().reset_index(name="count")
    )
    top_combos = combo_counts_final.head(5)
    logger.info("Top 5 most common label combinations:")
    logger.info("\n%s", top_combos.to_string(index=False))

    """
    logger.info("PHASE 3: Balancing based on individual label values...")
    plan_df_individual = get_individual_label_augmentation_plan(
        df_original=df,
        df_augmented=final_df,
        label_columns=label_columns,
        max_extra_aug_per_image_individual=3,  # or pull from config
    )

    for row in plan_df_individual.itertuples():
        logger.info(
            f"[INDIVIDUAL] Augmenting {row.filename} ({row.augmentations_needed}) due to {row.reason}"
        )
        img_path = os.path.join(image_directory, row.filename)
        for _ in range(row.augmentations_needed):
            combined_augmentation_for_image(
                img_path,
                output_directory,
                row.filename,
                df,
                num_augmented_images=1,
            )

    final_df = pd.read_excel(augmented_excel_file_path, sheet_name=augmented_sheet_name)
    log_dataset_summary(final_df, label_columns, "Final after individual-label balancing")

    original_count = df["filename"].nunique()
    final_count = final_df["filename"].nunique()
    new_images_created = final_count - original_count

    logger.info(f"Total new images created (including individual label balancing): {new_images_created}")

    growth_factor = final_count / original_count if original_count > 0 else 0
    logger.info(f"Dataset growth after individual-label balancing: {growth_factor:.2f}x")

    combo_counts_final = (
        final_df[label_columns].value_counts().reset_index(name="count")
    )
    top_combos = combo_counts_final.head(5)
    logger.info("Top 5 most common label combinations after individual-label balancing:")
    logger.info("\n%s", top_combos.to_string(index=False))
"""