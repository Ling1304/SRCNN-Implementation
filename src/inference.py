import cv2
import torch
import numpy as np
import os
from torchvision.transforms import ToTensor
from model import SRCNN
from utils import bgr2ycbcr, image2tensor, tensor2image, ycbcr2bgr

def pre_upscale_image(image_path, f1=9, f2=5, f3=5):
    """
    Function to upscale the image using bicubic interpolation so that after inference, 
    the output dimensions match the original image
    
    - Formula: (fsub - f1 - f2 - f3 + 3)
    """
    # Read the image
    image = cv2.imread(image_path)

    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate the upscaled size (fsub) required to recover the original dimensions
    upscale_h = h + f1 + f2 + f3 - 3
    upscale_w = w + f1 + f2 + f3 - 3

    # Resize the image using bicubic interpolation
    upscaled_image = cv2.resize(image, (upscale_w, upscale_h), interpolation=cv2.INTER_CUBIC)

    return upscaled_image

def inference(image_path, model_path, output_path, f1=9, f2=5, f3=5):
    """
    Perform inference using the trained SRCNN model.
    """
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load the trained model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode
    model.eval()

    # Pre-upscale the image
    upscaled_image = pre_upscale_image(image_path, f1, f2, f3)

    # Convert to float32 and normalize
    upscaled_image = upscaled_image.astype(np.float32) / 255.0

    # Convert to YCbCr and separate channels
    y_channel = bgr2ycbcr(upscaled_image, True)
    ycbcr_image = bgr2ycbcr(upscaled_image, False)
    _, cb_channel, cr_channel = cv2.split(ycbcr_image)

    # Convert Y channel to tensor
    y_tensor = image2tensor(y_channel, False, False).unsqueeze(0).to(
        device=device, memory_format=torch.channels_last, non_blocking=True
    )

    # Perform inference
    with torch.no_grad():
        sr_y_tensor = model(y_tensor).clamp_(0, 1.0)  # Shape: [1, 1, H_out, W_out]

    # Convert output tensor back to image
    sr_y_image = tensor2image(sr_y_tensor, False, False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0

    # Merge super-res Y channel with original Cb and Cr channels
    sr_ycbcr_image = cv2.merge([sr_y_image, cb_channel, cr_channel])
    sr_image = ycbcr2bgr(sr_ycbcr_image)

    # Save the super-resolved image
    cv2.imwrite(output_path, sr_image * 255.0)
    print(f"Super-resolved image saved to {output_path}")

if __name__ == "__main__":
    image_path = "C:/Users/Hezron Ling/Desktop/lr_comic.png"  # Path to input image
    model_path = "C:/Users/Hezron Ling/Desktop/SRCNN_model/SRCNN_model_2000.pth"  # Path to trained model
    output_path = "C:/Users/Hezron Ling/Desktop/lr_comic_x2.png"  # Path to save output image

    inference(image_path, model_path, output_path)
