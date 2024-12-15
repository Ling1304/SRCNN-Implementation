from math import log10, sqrt
import cv2
import numpy as np
import os

import torch

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

import torch

def save_model_state(model, epoch, model_path):
    """
    Function to save the trained model to be used for inference.
    """
    # Save the model every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Saving model at epoch {epoch + 1} to {model_path}...')
        torch.save(model.state_dict(), f'{model_path}/SRCNN_model_{epoch + 1}.pth')
        print(f'Model saved as SRCNN_model_{epoch + 1}.pth!')


