from math import log10, sqrt
import cv2
import numpy as np
import os

import torch
from torchvision.transforms import functional

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

# SOURCE: 
# - https://github.com/Lornatang/SRCNN-PyTorch/blob/main/imgproc.py#L326
# - https://stackoverflow.com/questions/54615539/different-color-conversion-from-rgb-to-ycbcr-with-opencv-and-matlab # OpenCV and Matlab uses different conversion rate
def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image

# https://github.com/Lornatang/SRCNN-PyTorch/blob/main/imgproc.py#L32
def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = functional.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor

# SOURCE: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/imgproc.py#L62
def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool):
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image

# SOURCE: https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py
def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image

