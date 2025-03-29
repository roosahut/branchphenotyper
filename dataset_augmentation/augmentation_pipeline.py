"""augmentation pipelines for different purposes """
import os
import pandas as pd
import random
import os.path
from os.path import splitext
from file_handling import get_image, save_image, read_excel_labels
from dataset_selection import select_images_for_augmentation
from transformations import (flip_image, zoom_in, rotate_image, increase_brightness, 
                              increase_contrast, adjust_saturation, blur_image,
                              warp_perspective,rgb_shift, channel_shuffle, convert_to_grayscale)

# FOR ONE IMAGE:

def augmentation_for_image(image_path, output_directory, img_filename, df):
    """Apply individual augmentations to one image and save each result"""
    img = get_image(image_path)
    labels = read_excel_labels(img_filename, df)
    
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
        "grayscale": convert_to_grayscale(img)
    }
    
    base_name, ext = splitext(img_filename)
    for suffix, transformed_img in transformations.items():
        new_filename = f"{base_name}_{suffix}{ext}"
        save_image(new_filename, transformed_img, output_directory, labels)

def combined_augmentation_for_image(image_path, output_directory, img_filename, df, num_combinations):
    """Apply combined augmentations to one image, which transfermations are done is randomly calculated"""
    img = get_image(image_path)
    labels = read_excel_labels(img_filename, df)
    
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
        "grayscale": convert_to_grayscale
    }
    
    base_name, ext = splitext(img_filename)
    for i in range(num_combinations):
        num_transformations = random.randint(2, len(available_transformations)) # min 2 transformatoins
        transformation_keys = random.sample(list(available_transformations.keys()), num_transformations)
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

def combined_augmentation_for_data_folder(image_directory, output_directory, excel_file_path, sheet_name, num_combinations, select_only_rare):
    """Creates num_combinations number of augmented images out of all images in the folder or only out of rare images"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    if select_only_rare:
        selected_images = select_images_for_augmentation(df)
    
    for img_filename in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_filename)
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if not select_only_rare or (select_only_rare and (img_filename in selected_images['filename'].values)):
                combined_augmentation_for_image(img_path, output_directory, img_filename, df, num_combinations=num_combinations)

def combined_augmentation_to_all_imgs_with_extra_for_rare(image_directory, output_directory, excel_file_path, sheet_name, baseline_num, extra_num):
    """Creates baseline number of augmented images out of all images in the folder and extra_num number of augmented images out of rare images"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    rare_df = select_images_for_augmentation(df)
    rare_filenames = set(rare_df['filename'].values)
  
    for img_filename in os.listdir(image_directory):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_directory, img_filename)
            # done for all images
            combined_augmentation_for_image(img_path, output_directory, img_filename, df, num_combinations=baseline_num)
            
            # extra augmentation for rare images
            if img_filename in rare_filenames:
                print("in rare images filenamse set" ,img_filename)
                combined_augmentation_for_image(img_path, output_directory, img_filename, df, num_combinations=extra_num)