""" all of the augmentaion functions to one image """
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

def flip_image(image):
    """Flips the image horizontally"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def zoom_in(image):
    """Zooms into the image by a random factor"""
    factor = random.uniform(1.1, 1.3) # change these to modify zoom level
    width, height = image.size
    new_width, new_height = int(width / factor), int(height / factor)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)

def rotate_image(image):
    """Rotates the image by a random angle"""
    angle = random.uniform(-15, 15) # change these to modify rotation, negative is to the left 
    return image.rotate(angle=angle, expand=True, fillcolor=(255, 255, 255))

def warp_perspective(image): # not sure if we want to use this, some images a bit funky
    """
    Applies a random perspective warp
    
    This function converts the PIL image to OpenCV format, randomly perturbs
    the corner points, applies the perspective transformation, and then converts back
    """
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols, _ = cv_img.shape
    src_pts = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]]) # corners
    max_shift_x = cols*0.1 # change these to impact shift level
    max_shift_y = rows*0.1 # change these to impact shift level
    dst_pts = np.float32([ #adding the random warp
        [random.uniform(0, max_shift_x), random.uniform(0, max_shift_y)],
        [cols - random.uniform(0, max_shift_x), random.uniform(0, max_shift_y)],
        [random.uniform(0, max_shift_x), rows - random.uniform(0, max_shift_y)],
        [cols - random.uniform(0, max_shift_x), rows - random.uniform(0, max_shift_y)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(cv_img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)

def rgb_shift(image):
    """Randomly shifts the intensity of each RGB channel independently"""
    np_img = np.array(image)
    shifts = np.random.randint(-30, 31, size=3)
    for i in range(3):
        np_img[..., i] = np.clip(np_img[..., i] + shifts[i], 0, 255)
    return Image.fromarray(np_img)

def channel_shuffle(image):
    """Randomly shuffles the RGB channels of the image"""
    np_img = np.array(image)
    channels = [0, 1, 2]
    random.shuffle(channels)
    shuffled = np_img[..., channels]
    return Image.fromarray(shuffled)

def increase_brightness(image):
    """Randomly increases or decreases brightness."""
    factor = random.uniform(0.8, 1.3) # change these to modify brightness 
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def increase_contrast(image):
    """Randomly increases or decreases contrast"""
    factor = random.uniform(0.8, 1.8) # change these to modify contrast
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image):
    """Randomly adjusts the saturation"""
    factor = random.uniform(0.8, 1.6) # change these to modify saturation
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def convert_to_grayscale(image):
    """Converts an image to grayscale and then back to RGB format"""
    return image.convert("L").convert("RGB")

def blur_image(image):
    """Applies a random level of Gaussian blur"""
    radius = random.uniform(1, 5) # change these to modify blur, found through manual testing: dont go much over 5
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

# image size TODO: uniform sizing that matches the backend->utils->preprocessing functions for the uploaded images?