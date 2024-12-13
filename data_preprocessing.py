import os
import cv2
import numpy as np

def get_sub_images(dataset_path, hr_output_path, lr_output_path, sub_image_size, stride, upscale_factor=2):
  # Ensure output directories exist
  os.makedirs(hr_output_path, exist_ok=True)
  os.makedirs(lr_output_path, exist_ok=True)

  # Get a list of image files from image folder path
  image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

  # For each image in image_file
  for image_file in image_files:
    # Read each image
    image = cv2.imread(image_file)

    # Get the height and width of image
    height, width, _ = image.shape

    sub_image_count = 0

    # Process the images to get the sub-images
    for h in range(0, height-sub_image_size+1, stride):
      for w in range(0, width-sub_image_size+1, stride):
        # Extract the HR sub-image
        hr_sub_image = image[h:h+sub_image_size, w:w+sub_image_size]

        # Save the HR sub-image
        hr_sub_image_name = f"t91_hr_{sub_image_count}.png"
        hr_sub_image_path = os.path.join(hr_output_path, hr_sub_image_name)
        cv2.imwrite(hr_sub_image_path, hr_sub_image)

        # Get the LR sub-image
        # - Blur with gaussian kernel
        blur_image = cv2.GaussianBlur(hr_sub_image, (3,3), 0)

        # - Downscale by upscaling factor
        downscale_size = (sub_image_size//upscale_factor, sub_image_size//upscale_factor)
        downscale_image = cv2.resize(blur_image, downscale_size, interpolation=cv2.INTER_LINEAR)

        # - Upscaling back to original size using bicubic interpolation
        upscale_image = cv2.resize(downscale_image, (sub_image_size, sub_image_size), interpolation=cv2.INTER_CUBIC)

        # Save the LR sub-image
        lr_sub_image_name = f"t91_lr_{sub_image_count}.png"
        lr_sub_image_path = os.path.join(lr_output_path, lr_sub_image_name)
        cv2.imwrite(lr_sub_image_path, upscale_image)

        sub_image_count += 1

dataset_path = '/content/drive/MyDrive/Colab Notebooks/SRCNN_Implementation/data/T91/Original'
hr_output_path = '/content/drive/MyDrive/Colab Notebooks/SRCNN_Implementation/data/T91/hr_sub_image '
lr_output_path = '/content/drive/MyDrive/Colab Notebooks/SRCNN_Implementation/data/T91/lr_sub_image'

get_sub_images(dataset_path,
               hr_output_path,
               lr_output_path,
               sub_image_size = 33,
               stride = 14,
               upscale_factor = 2)

