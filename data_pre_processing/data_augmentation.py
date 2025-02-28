import os
from PIL import Image, ImageEnhance, ImageFilter

image_directory = './images/original_images'
output_directory = './images/augmented_images'


#position augmentation
def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def zoom_in(image, factor=1.2):
    width, height = image.size
    new_width, new_height = int(width / factor), int(height / factor)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)

def zoom_out():
    #TODO
    pass

def rotate_image(image, angle):
    return image.rotate(angle=angle, expand=False, fillcolor=(255, 255, 255))

def shear_image(image, shear_factor):
    #not working as i want to, use a different library? this is leaving too much empty space in the photo
    #TODO: find a better way to do this
    width, height = image.size
    xshift = abs(shear_factor) * width
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE, (1, shear_factor, -xshift if shear_factor > 0 else 0, 0, 1, 0), Image.BICUBIC)



#color augmentation
def increase_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def increase_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

#blurring
def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5))

#image handling
def get_image(image_path):
    return Image.open(image_path)

def save_image(img_name, img, output_directory):
    img.save(os.path.join(output_directory, img_name))
    print(f"Augmented image '{img_name}' saved")

def augmentation_for_image(image_path, output_directory):
    img = get_image(image_path)
    
    # TODO: add randomness, generate values rather than set parameters
    flipped_img = flip_image(img)
    rotated_img_l = rotate_image(img, 15)
    rotated_img_r = rotate_image(img, 345)
    zoomed_in_img = zoom_in(img, 1.3)
    sheared_img = shear_image(img, 0.4)
    bright_img = increase_brightness(img, 1.2)
    contrast_img = increase_contrast(img, 1.5)
    saturated_img = adjust_saturation(img, 1.5)
    blurred_img = blur_image(img)

    combination_img = flip_image(img)
    combination_img = increase_contrast(combination_img, 1.5)
    combination_img = rotate_image(combination_img, 10)
    combination_img = zoom_in(combination_img, 1.2)
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    save_image(f"{name}_flipped{ext}", flipped_img, output_directory)
    save_image(f"{name}_rotated_l{ext}", rotated_img_l, output_directory)
    save_image(f"{name}_rotated_r{ext}", rotated_img_r, output_directory)
    save_image(f"{name}_zoomed_in_img{ext}", zoomed_in_img, output_directory)
    save_image(f"{name}_sheared{ext}", sheared_img, output_directory)
    save_image(f"{name}_bright{ext}", bright_img, output_directory)
    save_image(f"{name}_contrast{ext}", contrast_img, output_directory)
    save_image(f"{name}_saturated{ext}", saturated_img, output_directory)
    save_image(f"{name}_blurred{ext}", blurred_img, output_directory)

    save_image(f"{name}_combination_img{ext}", combination_img, output_directory)


def augmentation_for_data_folder(image_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #TODO: get excel row based on name of image, copy that images labels to another excel with the augmented name
    
    #TODO: choose the images that are not so commonly found in the dataset and run those several times with different parameters
    under_reprsented_image_types=[]

    #TODO: add combinations of augmentation techniques to one photo, add randomness to the process

    for img_filename in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_filename)
        if img_filename.endswith(('.png', '.jpg', '.jpeg')):
            augmentation_for_image(img_path, output_directory)


#run this with the correct parameters to do the augmentation for the entire folder
augmentation_for_data_folder(image_directory, output_directory)
