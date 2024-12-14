from math import log10, sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torchvision.utils import save_image

# SOURCE:
# - https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
# - https://pytorch.org/tutorials/beginner/saving_loading_models.html

def PSNR(hr_image, output, max_pixel=1):
  """
  Function that calculates Peak Signal to Noise Ratio (PSNR)
  - higher the value the better

  Equation:
  - MSE = (1 / N) * Σ (Pixel1 - Pixel2)^2
  - PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
  """
  # Convert PyTorch tensors to NumPy arrays
  hr_image = hr_image.cpu().detach().numpy()
  output = output.cpu().detach().numpy()

  # RMSE equation
  mse = np.mean((output - hr_image) ** 2)

  # RMSE is zero means no noise is present in the signal .
  if(mse == 0):
      return 100

  # PSNR equation
  else:
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def save_model_state(model, model_dir):
  """
  Function to save the trained model to be used for inference
  """
  # Ensure the model directory exists
  os.makedirs(model_dir, exist_ok=True)

  # Create full file path by appending 'model.pth' to the directory
  model_path = os.path.join(model_dir, 'SRCNN_model.pth')

  # Save the model
  print(f'Saving model to {model_path}...')
  torch.save(model.state_dict(), model_path)
  print('Model saved!')

