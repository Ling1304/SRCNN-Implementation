import os
import cv2
import numpy as np

import os
import cv2

def process_and_generate_sub_images(dataset_path, hr_output_path, lr_output_path, sub_image_size, stride, upscale_factor=2):
    """
    Process images and generate corresponding sub-images for both clear (HR) and blurred (LR) versions.
    """
    # Ensure output directories exist
    os.makedirs(hr_output_path, exist_ok=True)
    os.makedirs(lr_output_path, exist_ok=True)
    
    # Get a list of image files from the dataset path
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Read the original image (clear version)
        image = cv2.imread(image_file)
        height, width, _ = image.shape

        # Apply Gaussian blur to create the blurred version
        blur_image = cv2.GaussianBlur(image, (3, 3), 0)

        # Downscale and upscale the blurred image to create LR image
        downscale_size = (width // upscale_factor, height // upscale_factor)
        downscale_image = cv2.resize(blur_image, downscale_size, interpolation=cv2.INTER_LINEAR)
        upscale_image = cv2.resize(downscale_image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        sub_image_count = 0

        # Generate sub-images for both HR (clear) and LR (blurred)
        for h in range(0, height - sub_image_size + 1, stride):
            for w in range(0, width - sub_image_size + 1, stride):
                # Extract the HR (clear) sub-image
                hr_sub_image = image[h:h+sub_image_size, w:w+sub_image_size]
                hr_sub_image_name = f"{os.path.splitext(os.path.basename(image_file))[0]}_hr_{sub_image_count}.png"
                hr_sub_image_path = os.path.join(hr_output_path, hr_sub_image_name)
                cv2.imwrite(hr_sub_image_path, hr_sub_image)

                # Extract the LR (blurred) sub-image
                lr_sub_image = upscale_image[h:h+sub_image_size, w:w+sub_image_size]
                lr_sub_image_name = f"{os.path.splitext(os.path.basename(image_file))[0]}_lr_{sub_image_count}.png"
                lr_sub_image_path = os.path.join(lr_output_path, lr_sub_image_name)
                cv2.imwrite(lr_sub_image_path, lr_sub_image)

                sub_image_count += 1

def get_val_images(dataset_path, output_path, upscale_factor=3):
  '''
  Function to get low resolution images for validation (Set5).

  Validation set will consist of a low res image of the original image in set5.
  - using Gaussian Blur + Downscale + Upscale with Bicubic
  '''
  # Ensure output directories exist
  os.makedirs(output_path, exist_ok=True)

  # Get a list of image files from image folder path
  image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

  val_image_count = 0

  # For each image in image_file
  for image_file in image_files:
    # Read each image
    image = cv2.imread(image_file)

    # Get the height and width of image
    height, width, _ = image.shape

    # Get the LR validation image
    # - Blur with gaussian kernel
    blur_image = cv2.GaussianBlur(image, (3,3), 0)

    # - Downscale by upscaling factor
    downscale_size = (width//upscale_factor, height//upscale_factor)
    downscale_image = cv2.resize(blur_image, downscale_size, interpolation=cv2.INTER_LINEAR)

    # - Upscaling back to original size using bicubic interpolation
    upscale_image = cv2.resize(downscale_image, (width, height), interpolation=cv2.INTER_CUBIC) # Uncomment this if padding is used

    # # - Upscaling to 33x33 to ensure input matches output 21x21 during loss calculation in validation (due to no padding)
    # upscale_image = cv2.resize(downscale_image, (33, 33), interpolation=cv2.INTER_CUBIC) # Comment this if padding is used

    # Save the LR validation image
    original_name = os.path.basename(image_file)
    base_name, _ = os.path.splitext(original_name)
    image_name = f"lr_{base_name}.png"
    image_path = os.path.join(output_path, image_name)
    cv2.imwrite(image_path, upscale_image)

    val_image_count += 1

# dataset_path = "C:/Users/Hezron Ling/Desktop/data_SRCNN/Train(T91)/Original"
dataset_path = "C:/Users/Hezron Ling/Desktop/SRCNN-Implementation_(JUPYTER)/data/train"
hr_output_path = "C:/Users/Hezron Ling/Desktop/SRCNN-Implementation_(JUPYTER)/data/train/hr_sub_images"
lr_output_path = "C:/Users/Hezron Ling/Desktop/SRCNN-Implementation_(JUPYTER)/data/train/lr_sub_images"

sub_image_size = 33
stride = 14
upscale_factor = 2

process_and_generate_sub_images(dataset_path, hr_output_path, lr_output_path, sub_image_size, stride, upscale_factor)
print('Done!')

