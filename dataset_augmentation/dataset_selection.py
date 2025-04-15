"""
Provides functions to analyze a labeled dataset and determine which label combinations
are underrepresented. Based on this, it calculates how many augmented images are
needed to balance the dataset.

Main functions:
- determine_target_count: Calculates the desired frequency for each label combination
- get_augmentation_plan_by_combination: Identifies images to augment and how many times

Used as part of a pipeline to generate extra data for rare label configurations.
"""

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def log_dataset_summary(df, label_columns, stage_label=""):
    """
    Logs a summary of the dataset including:
    - Total rows
    - Unique filenames
    - Number of unique label combinations
    - Distribution of values in each label column (Top 30)

    Parameters:
        df (pd.DataFrame): DataFrame to summarize.
        label_columns (list): Columns that define the label structure.
        stage_label (str): Optional label for the stage (e.g., "Original", "Final").
    """

    logger.info(f"\n{stage_label} dataset summary:")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Unique filenames: {df['filename'].nunique()}")
    unique_combos = df[label_columns].drop_duplicates().shape[0]
    logger.info(f"Unique label combinations: {unique_combos}")

    for col in label_columns:
        logger.info(f"\n🔹 Distribution in '{col}' (Top {30}):")
        counts = df[col].value_counts(normalize=True).head(30)  # percent
        for val, pct in counts.items():
            count = df[col].value_counts()[val]
            bar = "█" * int(pct * 40)  # histogram bar
            logger.info(f"  {val}: {count} ({pct:.2%}) {bar}")


def determine_target_count(
    df, label_columns, method, value, baseline_augments_per_image=0
):
    """
    Calculates the desired number of images per label combination using a specified strategy.

    Parameters:
        df (pd.DataFrame): Full dataset including label columns.
        label_columns (list): Columns defining label combinations.
        method (str): Strategy to use — 'max', 'percentile', or 'fixed'.
        value (float or int): Parameter for the strategy (e.g., 0.8 for 80th percentile).
        baseline_augments_per_image (int): Number of augmentations applied to all images
                                           (to adjust rarity threshold accordingly).

    Returns:
        int: The calculated target count per label combination.
    """
    logger.info(
        f"Calculating target count for label combinations using strategy: {method}, baseline augmentations per image: {baseline_augments_per_image}"
    )

    counts = df[label_columns].value_counts()

    if method == "max":
        base_max = counts.max()
        target = base_max + (baseline_augments_per_image * base_max)
        return target
    elif method == "percentile":
        base_percentile = counts.quantile(value)
        target = int(base_percentile + (baseline_augments_per_image * base_percentile))
        return target
    elif method == "fixed":
        return int(value)
    else:
        raise ValueError("Unknown method for determining target count")


def get_augmentation_plan_by_combination_balanced(
    df_original, df_augmented, label_columns, target_count, max_extra_aug_per_image=4
):
    """
    Generates a per-image augmentation plan to balance underrepresented label combinations.

    For each label combination below `target_count`, determines how many new samples
    are needed, and distributes augmentation tasks across original images with that combo.

    Parameters:
        df_original (pd.DataFrame): Label data for original (unaugmented) images.
        df_augmented (pd.DataFrame): Label data after baseline augmentation.
        label_columns (list): Columns that define the label combinations.
        target_count (int): Desired total number of images per label combination.
        max_extra_aug_per_image (int): Optional cap on extra augmentations per image.

    Returns:
        pd.DataFrame: A DataFrame with:
            - 'filename': image to augment
            - 'augmentations_needed': how many extra times to augment
    """
    import pandas as pd

    logger.info("Computing augmentation plan to balance rare combinations...")

    logger.info("Counting label combinations after baseline augmentations...")
    combo_counts = df_augmented[label_columns].value_counts().reset_index(name="count")
    underrepresented = combo_counts[combo_counts["count"] < target_count].drop(
        columns="count"
    )

    logger.info("Underrepresented label combinations:")
    logger.info("\n%s", underrepresented.to_string(index=False))

    logger.info(
        "Finding original filenames for images with underrepresented label combinations..."
    )
    rare_originals = pd.merge(
        df_original, underrepresented, on=label_columns, how="inner"
    )

    logger.info(
        "Distributing total needed image generation for specific label type to different original images"
    )
    plan_rows = []
    for _, group in rare_originals.groupby(label_columns):
        combo_dict = group.iloc[0][label_columns].to_dict()

        current_total = df_augmented[
            (df_augmented[label_columns] == group.iloc[0][label_columns].values).all(
                axis=1
            )
        ].shape[0]

        if current_total >= target_count:
            logger.info(
                f"Skipping combo {combo_dict} — already has {current_total} images"
            )
            continue

        needed = target_count - current_total
        count = group.shape[0]

        per_image = needed // count
        leftover = needed % count
        logger.info(
            f"Need {needed} images generated to balance label distribution, generating {per_image} per image, no of images with this label distribution: {count} leftover: {leftover}."
        )

        for i, (_, row) in enumerate(group.iterrows()):
            aug_count = per_image + (1 if i < leftover else 0)
            if aug_count > 0:
                plan_rows.append(
                    {
                        "filename": row["filename"],
                        "augmentations_needed": min(aug_count, max_extra_aug_per_image),
                    }
                )

    plan_df = pd.DataFrame(plan_rows)

    logger.info(f"Final rare image augmentation plan ({len(plan_df)} rows):")
    if not plan_df.empty:
        logger.info(plan_df.to_string(index=False))

    return plan_df


import numpy as np

def get_individual_label_augmentation_plan(
    df_original,
    df_augmented,
    label_columns,
    max_extra_aug_per_image_individual,
):
    """
    For each label column, calculate how underrepresented each value is.
    If there's a large gap between the max and current count, schedule extra augmentations.
    """
    plan_rows = []
    for col in label_columns:
        value_counts = df_augmented[col].value_counts()
        max_count = value_counts.max()

        for val, count in value_counts.items():
            gap = max_count - count
            if gap < 1:
                continue

            df_with_val = df_original[df_original[col] == val]
            if df_with_val.empty:
                continue

            per_image = max(1, gap // len(df_with_val))
            per_image = min(per_image, max_extra_aug_per_image_individual)

            leftover = gap % len(df_with_val)
            for i, row in enumerate(df_with_val.itertuples()):
                aug_count = per_image + (1 if i < leftover else 0)
                if aug_count > 0:
                    plan_rows.append({
                        "filename": row.filename,
                        "augmentations_needed": aug_count,
                        "reason": f"{col}={val}"
                    })

    plan_df = pd.DataFrame(plan_rows)
    logger.info(f"Generated individual label augmentation plan with {len(plan_df)} entries.")
    if not plan_df.empty:
        logger.info(plan_df.groupby('reason').size().to_string())
    return plan_df
