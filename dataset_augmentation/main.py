""" Run different augmentation pipelines """

from config import image_directory, output_directory, excel_file_path, sheet_name, image_filename, os
from augmentation_pipeline import (combined_augmentation_for_data_folder, augmentation_for_image, combined_augmentation_to_all_imgs_with_extra_for_rare)
from file_handling import get_image
from transformations import rotate_image
import pandas as pd
import time

def run_combined_augmentation_to_all_imgs_with_extra_for_rare(image_directory, output_directory, excel_file_path, sheet_name, baseline_num, extra_num):
    """
    Runs combined augmentation on the entire dataset and creates extra augmented images for rare label types.
    This function loads the phenotype excel label information and updates the augmented images phenotype excel file accordingly

    This function applies the combined augmentation pipeline to all images and creates:
      - baseline_num augmented versions for every image, and
      - extra_num additional augmented versions for images that are selected as "rare" based on label criteria.
    
    Parameters:
        image_directory (str): Path to the folder containing original images.
        output_directory (str): Path to the folder where augmented images will be saved.
        excel_file_path (str): Path to the Excel file with phenotype labels.
        sheet_name (str): The Excel sheet name with label data.
        baseline_num (int): Number of augmented versions to create for every image.
        extra_num (int): Number of additional augmented versions to create for rare images.
    
    """
    print("Running combined augmentation (chaining random transformations) on the entire dataset and creates extra augmented images for rare label types...")
    combined_augmentation_to_all_imgs_with_extra_for_rare(image_directory, output_directory, excel_file_path, sheet_name, baseline_num, extra_num)

  
def run_combined_augmentations_for_folder_with_rare_selection(num_combinations,select_only_rare):
    """
    Runs combined augmentation (chaining random transformations) on the entire dataset.
    This function loads the phenotype excel label information and updates the augmented images phenotype excel file accordingly

    Depending on the parameter, this function either augments:
      - only the rare images (if select_only_rare is True), or 
      - all images in the folder (if select_only_rare is False).
    
    Parameters:
        num_combinations (int): Number of augmented versions to create per original image.
        select_only_rare (bool): If True, only images meeting the rarity criteria will be augmented;
                                 if False, all images will be processed.
    """
    print("Running combined augmentation (chaining random transformations) on either entire dataset or only on the rare label images...")
    combined_augmentation_for_data_folder(image_directory, output_directory, excel_file_path, sheet_name, num_combinations, select_only_rare)


    
def run_all_augmentations_for_one_image(image_path, image_filename):
    """
    Runs all predefined individual augmentation functions on a single image.
    
    This function loads the Excel label information, applies all individual augmentations
    to the image (each resulting in a separate saved image), and updates the Excel file accordingly.
    
    Parameters:
        image_path (str): Full path to the image.
        image_filename (str): The filename of the image.
    """
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    print(f"Running all individual augmentations on image: {image_filename}")
    augmentation_for_image(image_path, output_directory, image_filename, df)


def run_one_augmentation_for_one_image():
    """
    Applies a single transformation (rotate) to a chosen image for exploration purposes.
    This function loads an image, applies the 'rotate' transformation, and displays the result.
    This functions does NOT read or modify any excel files.
    """
    print("\nApplying a single transformation for exploration purposes...")
    image=get_image(image_path='./images/original_images/B-1.jpg')
    transformed_image=rotate_image(image)
    transformed_image.show()  # This will open the image in your default image viewer

def delete_augmented_images_folder_contents():
    """Deletes all files in the augmented images output directory"""
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        os.remove(file_path)


"""
Executing of pipelines from here, uncomment what you want to run here or call these functions like this from anywhere

Most likely the run_combined_augmentation_to_all_imgs_with_extra_for_rare is the one to use when we want to actually 
create new images for the dataset but for testing and figuring out parameters and augmentation ranges and frequencies
these other functions are here to help with that.

Also added time functions to print out how long it took to run the chosen pipeline
"""

start_time = time.time()

#run_combined_augmentation_to_all_imgs_with_extra_for_rare(image_directory, output_directory, excel_file_path, sheet_name, 1, 2)
#run_combined_augmentations_for_folder_with_rare_selection(num_combinations=2, select_only_rare=True)
#run_all_augmentations_for_one_image(image_directory+"/"+image_filename, image_filename)
#run_one_augmentation_for_one_image()

end_time = time.time()
duration = end_time - start_time

print(f"pipeline ran for {duration:.2f} seconds or {duration/60:.2f} minutes")

#delete_augmented_images_folder_contents()

