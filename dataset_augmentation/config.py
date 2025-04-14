import os

# Image to use for testing individual augmentations, not needed otherwise
image_filename = "B-1.jpg"

# Directory paths
image_directory = "./images/original_images"  # Path to original input images
output_directory = "./images/augmented_images"  # Output path for augmented images

# Excel configuration
excel_file_path = "./labels/phenotype_labels.xlsx"  # Path to original labels Excel
augmented_excel_file_path = (
    "./labels/augmented_images_phenotype_labels.xlsx"  # Path to augmented labels Excel
)
sheet_name = "birch_labels_final"  # Sheet containing label data
augmented_sheet_name = "augmented_images_labels"

# Number of augmented versions to create for each image, this is for increasing the entire dataset
num_augmented_images = 1

# Strategy to define rarity of a label combination:
# - 'fixed': target_value is a fixed count (e.g., 3 means rare = < 3 occurrences of that label combination).
# - 'percentile': target_value is the quantile cutoff (e.g., 0.9 = bottom 10% combos)
# - 'max': target_value is not needed, this balances the entire dataset. Takes the max count of label combinations and creates max_count-1 amount of augmented images from the rest of the dataset. End result is balanced dataset.
target_count_strategy = "fixed"
target_value = 5

# Cap on how many extra images we generate for rare label combinations
max_extra_aug_per_image = 4
