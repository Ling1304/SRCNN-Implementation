import torch
import numpy as np
import cv2
import os
from utils import bgr2ycbcr, image2tensor

import torch
from torch.utils.data import Dataset, DataLoader

# SOURCES:
# - https://pytorch.org/docs/stable/data.html
# - https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
# - https://pytorch.org/docs/stable/tensors.html
# - https://en.wikipedia.org/wiki/YCbCr

class SRCNN_Dataset(Dataset):
  def __init__(self, lr_dir, hr_dir):
    # Set LR and HR paths
    self.lr_dir = lr_dir
    self.hr_dir = hr_dir

    # Get a list of images in LR and HR path, then sort to ensure LR images correspond to HR images
    self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
  
  def __len__(self): 
    return (len(self.lr_files))

  def __getitem__(self, index): 
    """
    Function to get the Y channel in YCbCr color space
    """
    # Get the LR and HR image path
    lr_file = os.path.join(self.lr_dir, self.lr_files[index])
    hr_file = os.path.join(self.hr_dir, self.hr_files[index])

    # Read each LR and HR image as float32, then normalize it to 0-1
    lr_image = cv2.imread(lr_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    hr_image = cv2.imread(hr_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255

    # Convert to YCbCr color space then get Y channel 
    lr_y_image = bgr2ycbcr(lr_image, only_use_y_channel=True)
    hr_y_image = bgr2ycbcr(hr_image, only_use_y_channel=True)
    
    # # Resize HR image to 21x21 to match model output (  due to no padding, if padding is used, comment this)
    # hr_image_y = cv2.resize(hr_image_y, (21, 21), interpolation=cv2.INTER_CUBIC)

    # Convert image data into Tensor stream format (PyTorch).
    # Note: The range of input and output is between [0, 1]
    lr_y_tensor = image2tensor(lr_y_image, range_norm=False, half=False)
    hr_y_tensor = image2tensor(hr_y_image, range_norm=False, half=False)

    # Return the tensors
    return(lr_y_tensor, hr_y_tensor)

def load_datasets(train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir):
  """
  Function to load the training and testing dataset
  Note: 'DataLoader' automatically loops through data to load them
  """
  # Get the training and validation data in PyTorch tensors
  data_train = SRCNN_Dataset(train_lr_dir, train_hr_dir)
  data_val = SRCNN_Dataset(val_lr_dir, val_hr_dir)

  # Load the training and validation data using 'DataLoader'
  train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
  val_loader = DataLoader(data_val, batch_size=1)

  return train_loader, val_loader
