import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from model import SRCNN
from utils import bgr2ycbcr, image2tensor, tensor2image, ycbcr2bgr

# Define inference function
def inference(image_path, model_path, output_path):
    """
    Function to perform inference using the trained SRCNN model with padding.
    """
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load the trained model 
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Set model to evaluation mode
    model.eval()

    # Read the input image as float32 and normalize it
    lr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # Convert to ycbcr to get y channel of lr image
    lr_y_image = bgr2ycbcr(lr_image, True)

    # Convert to ycbcr to get cbcr channel of lr image
    lr_ycbcr_image = bgr2ycbcr(lr_image, False)
    _, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
    
    # Convert y channel image to tensor
    lr_y_tensor = image2tensor(lr_y_image, False, False).unsqueeze_(0)

    # Load the tensor in device
    lr_y_tensor = lr_y_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    # Perform inference, use clamp_ so that output tensor is normalized
    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)  # Shape: [1, 1, H_out, W_out]

    # Convert y channel tensor to image
    sr_y_image = tensor2image(sr_y_tensor, False, False)

    # Normalize the y channel image
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    
    # Combines super res y channel image with the low res cbcr image
    sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
    sr_image = ycbcr2bgr(sr_ycbcr_image)

    # Save image
    cv2.imwrite(output_path, sr_image * 255.0)

    print(f"SR image save to {output_path}")

if __name__ == "__main__":
    image_path = "C:/Users/Hezron Ling/Desktop/output1.png"  # Path to input image
    model_path = "C:/Users/Hezron Ling/Desktop/SRCNN_model/SRCNN_model_2000.pth"  # Path to trained model
    output_path = "C:/Users/Hezron Ling/Desktop/output_SR_image.png"  # Path to save output image

    inference(image_path, model_path, output_path)
